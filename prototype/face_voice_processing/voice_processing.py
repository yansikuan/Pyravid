from threading import Lock
import json, os, io, subprocess, argparse, uuid, base64
from tempfile import NamedTemporaryFile

import torch, torchaudio
from pydub import AudioSegment
from speakerlab.process.processor import FBank
from speakerlab.utils.builder import dynamic_import
from prototype.tools.utils import parse_timestamp_to_seconds, save_json

pretrained_state = torch.load(
    "./models/pretrained_eres2netv2.ckpt",
    map_location="cpu",
)
model = {
    "obj": "speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2",
    "args": {"feat_dim": 80, "embedding_size": 192},
}
global_voice_id_counter = 0
voice_id_lock = Lock()
embedding_model = dynamic_import(model["obj"])(**model["args"])
embedding_model.load_state_dict(pretrained_state)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model.to(device)
embedding_model.eval()
feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

def extract_audio_from_video(video_path: str, sample_rate: int = 16000) -> AudioSegment:
    """Extracts mono audio from a video file as AudioSegment."""
    with NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-f",
            "wav",
            tmp_audio.name,
            "-y",
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        audio = AudioSegment.from_wav(tmp_audio.name)
    return audio

def reset_voice_id_counter():
    global global_voice_id_counter
    with voice_id_lock:
        global_voice_id_counter = 0

def get_next_voice_id():
    global global_voice_id_counter
    with voice_id_lock:
        voice_id = f"voice_{global_voice_id_counter}"
        global_voice_id_counter += 1
        return voice_id

def compute_embedding_from_wav_bytes(wav_bytes: bytes):
    """Computes speaker embedding from raw wav bytes."""
    wav_io = io.BytesIO(wav_bytes)
    wav, fs = torchaudio.load(wav_io)
    if fs != 16000:
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            wav, fs, effects=[["rate", "16000"]]
        )
    if wav.shape[0] > 1:
        wav = wav[0, :].unsqueeze(0)
    feat = feature_extractor(wav).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embedding_model(feat).detach().squeeze(0).cpu().numpy()
    return emb

def save_voice_embedding_record(
    fact_id: str,
    output_dir: str,
    video_name: str,
    clip_name: str,
    asr_text: str,
    start_time: float,
    end_time: float,
    wav_bytes: bytes,
    embedding_vector,
):
    """Save one embedding record as JSON, including audio base64."""
    os.makedirs(output_dir, exist_ok=True)
    vid = get_next_voice_id()

    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

    record = {
        "fact_id": fact_id,
        "voice_id": vid,
        "video_name": video_name,
        "clip_name": clip_name,
        "asr_text": asr_text,
        "start_time": start_time,
        "end_time": end_time,
        "audio_base64": audio_b64,
        "voice_embedding": embedding_vector.tolist(),
    }

    out_path = os.path.join(output_dir, f"{vid}.json")
    save_json(record, out_path)
    return out_path

def process_facts_to_voice_embeddings(
    video_folder: str,
    facts_path: str,
    output_dir: str,
    sample_rate: int = 16000,
    reset_counter: bool = True,
):

    if reset_counter:
        reset_voice_id_counter()
    video_folder = os.path.expanduser(video_folder)
    facts_path = os.path.expanduser(facts_path)
    output_dir = os.path.expanduser(output_dir)

    with open(facts_path, "r", encoding="utf-8") as f:
        facts_json = json.load(f)

    for clip_id, clip_data in facts_json.items():
        video_path = os.path.join(video_folder, f"{clip_id}.mp4")
        if not os.path.exists(video_path):
            continue

        audio = extract_audio_from_video(video_path, sample_rate=sample_rate)
        video_name = video_folder.split("/")[-1]

        facts = clip_data.get("facts", [])
        for fact in facts:
            fact_id = fact.get("id")
            asr_periods = fact.get("asr_periods", [])
            if not asr_periods:
                continue

            for period in asr_periods:
                start =parse_timestamp_to_seconds(period.get("starttime")) - int(clip_id) * 30
                end = parse_timestamp_to_seconds(period.get("endtime")) - int(clip_id) * 30
                text = period.get("text", "")

                if start is None or end is None or end < start:
                    continue

                start_ms = int(start * 1000)
                end_ms = int(end * 1000)
                if start_ms < 0 or end_ms > len(audio):
                    continue

                segment = audio[start_ms:end_ms]
                buf = io.BytesIO()
                segment.export(buf, format="wav")
                wav_bytes = buf.getvalue()

                emb = compute_embedding_from_wav_bytes(wav_bytes)

                save_voice_embedding_record(
                    fact_id=fact_id,
                    output_dir=output_dir,
                    video_name=video_name,
                    clip_name=clip_id,
                    asr_text=text,
                    start_time=period.get("starttime"),
                    end_time=period.get("endtime"),
                    wav_bytes=wav_bytes,
                    embedding_vector=emb,
                )

def process_clip_voices_online(
    clip_id,
    clip_data,
    video_folder,
    voices_dir,
    sample_rate=16000,
    video_name=None,
    reset_counter=False,
):

    if reset_counter:
        reset_voice_id_counter()

    video_path = os.path.join(video_folder, f"{clip_id}.mp4")
    if not os.path.exists(video_path):
        print(f"[Voice] Clip {clip_id}: video not found: {video_path}")
        return

    audio = extract_audio_from_video(video_path, sample_rate=sample_rate)
    facts = clip_data.get("facts", [])
    if not facts:
        print(f"[Voice] Clip {clip_id}: no facts, skip.")
        return

    voices_dir_for_clip = os.path.join(voices_dir, video_name)
    os.makedirs(voices_dir_for_clip, exist_ok=True)
    voices_dir_for_clip = os.path.join(voices_dir_for_clip, str(clip_id))
    print(f"[Voice] Clip {clip_id}: voices dir for clip: {voices_dir_for_clip}")
    os.makedirs(voices_dir_for_clip, exist_ok=True)


    import io

    for fact in facts:
        fact_id = fact.get("id")
        asr_periods = fact.get("asr_periods", [])
        if not asr_periods:
            continue

        for period in asr_periods:
            start = parse_timestamp_to_seconds(period.get("starttime")) - int(clip_id) * 30
            end = parse_timestamp_to_seconds(period.get("endtime")) - int(clip_id) * 30
            text = period.get("text", "")

            if start is None or end is None or end <= start:
                continue

            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            if start_ms < 0 or end_ms > len(audio):
                continue

            segment = audio[start_ms:end_ms]
            buf = io.BytesIO()
            segment.export(buf, format="wav")
            wav_bytes = buf.getvalue()

            emb = compute_embedding_from_wav_bytes(wav_bytes)

            save_voice_embedding_record(
                fact_id=fact_id,
                output_dir=voices_dir_for_clip,
                video_name=video_name,
                clip_name=str(clip_id),
                asr_text=text,
                start_time=period.get("starttime"),
                end_time=period.get("endtime"),
                wav_bytes=wav_bytes,
                embedding_vector=emb,
            )

    print(f"[Voice] Clip {clip_id}: voice segments processed.")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate per-ASR-segment voice embedding files from facts.json"
    )
    parser.add_argument("--video-folder", default="/path/to/external/m3-agent-videomme/data/clips/_cZXyj6rYVg", help="Folder with input videos.")
    parser.add_argument("--facts-path", default="../test_with_timestamp/_cZXyj6rYVg.json", help="Path to facts.json.")
    parser.add_argument("--output-dir", default="../voices", help="Folder to save JSON embeddings.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    process_facts_to_voice_embeddings(
        video_folder=args.video_folder,
        facts_path=args.facts_path,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
