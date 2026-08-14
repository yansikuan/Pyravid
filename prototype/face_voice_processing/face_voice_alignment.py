import os
import json

from prototype.tools.prompts import FACE_VOICE_ALIGNMENT_FACT_LEVEL
from prototype.tools.utils import (
    call_model,
    collect_faces_by_clip,
    collect_voices_by_clip,
    load_facts,
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

def process_clip_equivalence_online(
    clip_id,
    captions_path,
    faces_dir,
    voices_dir,
    equivalence_dir,
):

    captions_all = load_facts(captions_path)
    video_name = os.path.basename(os.path.dirname(captions_path))
    out_path = os.path.join(equivalence_dir, video_name, str(clip_id) + ".json")

    voices_dir_for_video = os.path.join(voices_dir, video_name, str(clip_id))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        eq_all = load_facts(out_path)
    else:
        eq_all = {}

    clip_content = captions_all.get(str(clip_id), {})
    if not clip_content:
        print(f"[Align] Clip {clip_id}: no caption content, skip alignment.")
        return

    model_out = clip_content.get("model_output", {})
    summary = model_out.get("character_level_summary", "")
    descriptions = []

    if "facts" in model_out:
        for fact_id, fact_info in model_out["facts"].items():
            grounded = fact_info.get("character_level_facts")
            if grounded:
                descriptions.append(f"[{fact_id}] {grounded}")
            for cid, cinfo in fact_info.get("character_details", {}).items():
                for key in ["appearance", "actions", "speech", "role", "relation"]:
                    val = cinfo.get(key)
                    if val and isinstance(val, str):
                        descriptions.append(f"{cid}: {val}")

    faces_idx = collect_faces_by_clip(faces_dir)
    voices_idx = collect_voices_by_clip(voices_dir_for_video)

    faces_here = faces_idx.get(str(clip_id), [])
    voices_here = voices_idx.get(str(clip_id), [])

    inputs = []
    if summary:
        inputs.append({"type": "text", "content": f"summary: {summary}"})

    if descriptions:
        inputs.append({"type": "text", "content": "\n".join(descriptions)})

    if faces_here:
        face_pairs = [(f"<{f['face_id']}> from <{f['fact_id']}>", f["image_base64"]) for f in faces_here]
        inputs.append({"type": "images/jpeg", "content": face_pairs})

        fact_map_lines = [
            f"<{f['face_id']}> belongs to <{f['fact_id']}>"
            for f in faces_here if f.get("fact_id")
        ]
        if fact_map_lines:
            inputs.append({"type": "text", "content": "\n".join(fact_map_lines)})
    else:
        inputs.append({"type": "text", "content": "No faces detected in this clip."})

    if voices_here:
        for v in voices_here:
            seg_text = f"<{v['voice_id']}> [{v['start_time']}-{v['end_time']}]: {v.get('asr','')}"
            inputs.append({"type": "text", "content": seg_text})
            if v.get("audio_base64"):
                inputs.append({"type": "audio_base64/wav", "content": v["audio_base64"]})

    model_resp = call_model(inputs, system_prompt=FACE_VOICE_ALIGNMENT_FACT_LEVEL)
    try:
        parsed = smart_json_loads(model_resp)
    except Exception as e:
        print(f"[Align] Clip {clip_id}: JSON parse failed, keep raw output. ({e})")
        parsed = {"raw_output": model_resp}

    eq_all[str(clip_id)] = parsed

    os.makedirs(equivalence_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump(eq_all, wf, indent=2, ensure_ascii=False)

    print(f"[Align] Clip {clip_id}: updated equivalence file {out_path}")
