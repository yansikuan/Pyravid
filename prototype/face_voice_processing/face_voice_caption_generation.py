import os
import json
import base64
from typing import Dict, List, Any, Tuple

from prototype.tools.prompts import FACT_LEVEL_CAPTION_GENERAION_V4
from prototype.tools.utils import (
    call_model,
    collect_faces_by_clip,
    collect_voices_by_clip,
    smart_json_loads,
)


def load_b64_from_txt(b64_path: str) -> str:
    if not b64_path or not os.path.exists(b64_path):
        return ""
    try:
        with open(b64_path, "r", encoding="utf-8") as f:
            data = f.read().strip()
        if data.startswith("data:image"):
            data = data.split("base64,", 1)[-1]
        return data.replace("\n", "").replace("\r", "").replace(" ", "")
    except Exception as e:
        print(f"[⚠️] Failed to read {b64_path}: {e}")
        return ""

def load_video_as_base64(video_path: str) -> str:
    if not os.path.exists(video_path):
        print(f"[⚠️] Video not found: {video_path}")
        return ""
    try:
        with open(video_path, "rb") as f:
            data = f.read()
        b64_data = base64.b64encode(data).decode("utf-8")
        print(f"[🎥] Loaded video: {os.path.basename(video_path)} ({len(data)/1e6:.2f} MB)")
        return b64_data
    except Exception as e:
        print(f"[⚠️] Failed to read video {video_path}: {e}")
        return ""

def aggregate_multimodal_inputs(
    clip_id: str,
    clip_data: Dict[str, Any],
    face_idx: Dict[str, Dict[str, List[Dict[str, Any]]]],
    voice_idx: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Tuple[str, List[Tuple[str, str]], List[Tuple[str, str]], List[str], List[Tuple[str, str]], List[str], List[str]]:

    texts = []
    keyframe_pairs: List[Tuple[str, str]] = []
    face_pairs: List[Tuple[str, str]] = []
    voice_texts: List[str] = []
    voice_audio_pairs: List[Tuple[str, str]] = []

    for fact in clip_data.get("facts", []):
        fact_id = str(fact.get("id"))
        desc = fact.get("description") or fact.get("summary") or ""
        names = fact.get("name_mentions", [])
        if desc and names:
            names_str = ", ".join(names)
            texts.append(f"[{fact_id}] (names mentioned: {names_str}) {desc}")

        for k in fact.get("key_frames", []):
            b64_path = k.get("b64_path")
            b64_str = load_b64_from_txt(b64_path)
            if b64_str:
                keyframe_pairs.append((f"Keyframes from <{fact_id}>", b64_str))

    current_clip_faces = face_idx.get(str(clip_id), [])
    print(f"  📸 Clip {clip_id}: found {len(current_clip_faces)} face entries")

        # faces
    for f in current_clip_faces:
        fid = f.get("face_id")
        fact_id = f.get("fact_id")
        img_b64 = f.get("image_base64")
        if fid and img_b64:
            face_pairs.append((f"<{fid}> from <{fact_id}>", img_b64))

    current_clip_voices = voice_idx.get(str(clip_id), [])
    print(f"  🔊 Clip {clip_id}: found {len(current_clip_voices)} voice entries")
        # voices
    for v in current_clip_voices:
        vid = v.get("voice_id")
        if vid:
            seg_text = f"<{vid}> [{v.get('start_time')}-{v.get('end_time')}]: {v.get('asr','')}"
            voice_texts.append(seg_text)
            audio_str = v.get("audio_base64", "")
            if isinstance(audio_str, str) and len(audio_str) > 0:
                voice_audio_pairs.append(("audio_base64/wav", audio_str))

    text_blob = "\n".join(texts)
    return text_blob, keyframe_pairs, face_pairs, voice_texts, voice_audio_pairs

def process_clip_captions_online(
    clip_id,
    clip_data,
    facts_path,
    faces_dir,
    voices_dir,
    captions_dir,
    video_folder,
):

    video_name = os.path.splitext(os.path.basename(facts_path))[0]
    out_path = os.path.join(captions_dir, video_name, f"{clip_id}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    voices_dir_for_video = os.path.join(voices_dir, video_name, str(clip_id))

    face_idx = collect_faces_by_clip(faces_dir)
    voice_idx = collect_voices_by_clip(voices_dir_for_video)

    (
        text_blob,
        keyframe_pairs,
        face_pairs,
        voice_texts,
        voice_audio_pairs,
    ) = aggregate_multimodal_inputs(
        clip_id=str(clip_id),
        clip_data=clip_data,
        face_idx=face_idx,
        voice_idx=voice_idx,
    )

    print(f"[Caption] Clip {clip_id}: faces={len(face_pairs)}, voices={len(voice_audio_pairs)}")

    video_path = os.path.join(video_folder, f"{clip_id}.mp4")
    video_b64 = ""
    if os.path.exists(video_path):
        with open(video_path, "rb") as f:
            import base64
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

    inputs = []

    if text_blob:
        inputs.append({"type": "text", "content": text_blob})
    if video_b64:
        inputs.append({"type": "video_base64/mp4", "content": video_b64})
    if keyframe_pairs:
        inputs.append({"type": "images/jpeg", "content": keyframe_pairs})
    if face_pairs:
        inputs.append({"type": "images/jpeg", "content": face_pairs})
    else:
        inputs.append({"type": "text", "content": "No faces detected in this clip."})
    for t in voice_texts:
        inputs.append({"type": "text", "content": t})
    for _, audio in voice_audio_pairs:
        inputs.append({"type": "audio_base64/wav", "content": audio})

    model_resp = call_model(inputs, system_prompt=FACT_LEVEL_CAPTION_GENERAION_V4)
    try:
        parsed = smart_json_loads(model_resp)
    except Exception:
        parsed = {"raw_output": model_resp}

    captions_all = {str(clip_id): {"model_output": parsed}}

    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump(captions_all, wf, indent=2, ensure_ascii=False)

    print(f"[Caption] Clip {clip_id}: updated captions file {out_path}")
