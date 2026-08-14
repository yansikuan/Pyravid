import os
import io
import re
import json
import time
import base64
import argparse
import traceback
from PIL import Image
from typing import List
from collections import defaultdict
from moviepy import VideoFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from prototype.tools.prompts import FACTUAL_EXTRACTION_PROMPT_V5_2
from prototype.tools.api_client import APIClient
from prototype.tools.utils import (
    generate_messages,
    parse_timestamp_to_seconds,
    seconds_to_timestamp,
    smart_json_loads,
)

# ================== Global rate limiter ==================
rate_lock = Lock()
last_call_time = 0
MIN_INTERVAL = 1.0  # 1 request/sec → 60 RPM safe for Gemini key

def rate_limited_call(func, *args, **kwargs):
    global last_call_time
    with rate_lock:
        now = time.time()
        delta = now - last_call_time
        if delta < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - delta)
        last_call_time = time.time()
    return func(*args, **kwargs)

# ================== Utilities ==================
def encode_video_b64(path):
    with open(path, "rb") as vf:
        return base64.b64encode(vf.read()).decode("utf-8")

def get_frame_ids_from_facts(response: str) -> List[int]:
    json_response = smart_json_loads(response)
    frames = []
    for fact in json_response.get("facts", []):
        timestamp = fact.get("timestamp", "")
        frames.append(parse_timestamp_to_seconds(timestamp))
    return frames


def extract_frames_and_save(video_path: str, frame_ids: List[float], output_folder: str, adjusted_ids: dict = None) -> dict:
    results = defaultdict(list)

    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    clip = VideoFileClip(video_path)
    adjusted_ids = adjusted_ids or {}

    try:
        for frame_id in sorted(set(frame_ids)):
            if frame_id < 0 or frame_id > clip.duration:
                print(f"[!] Frame {frame_id}s out of range for {video_path} (duration: {clip.duration}s)")
                continue

            frame = clip.get_frame(frame_id)
            img = Image.fromarray(frame.astype("uint8"))

            adjusted_id = adjusted_ids.get(frame_id, frame_id)
            fname = f"frame_{adjusted_id}"

            jpeg_path = os.path.abspath(os.path.join(output_folder, f"{fname}.jpg"))
            b64_path = os.path.abspath(os.path.join(output_folder, f"{fname}.b64.txt"))

            img.save(jpeg_path, format="JPEG", quality=85)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

            with open(b64_path, "w") as f:
                f.write(f"data:image/jpeg;base64,{b64_data}")

            results[adjusted_id].append({
                "jpg_path": jpeg_path,
                "b64_path": b64_path
            })
    finally:
        clip.close()

    return results

def extract_visual_facts_from_video(video_path: str, key_path: str) -> str:
    """Call Qwen API with rate limiting and retries"""
    client = APIClient(api="gemini", key_path=key_path, model="gemini-2.5-flash", embedding_model="")
    video_b64 = encode_video_b64(video_path)
    inputs = [{"type": "video_base64/mp4", "content": video_b64}]
    messages = generate_messages(inputs, system_prompt=FACTUAL_EXTRACTION_PROMPT_V5_2)

    # 3 retries for safety
    for attempt in range(3):
        try:
            response = rate_limited_call(
                client.obtain_response,
                messages=messages,
                max_tokens=20000,
                temperature=0.0
            )
            if response:
                return response
        except Exception as e:
            print(f"[!] Attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    raise RuntimeError(f"Failed to extract facts for {video_path} after 3 retries.")


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


# ================== Worker ==================
def add_fact_ids_to_facts_json(facts_json: dict) -> dict:
    facts = facts_json.get("facts", [])
    for fact_id, fact in enumerate(facts):
        fact["id"] = f"fact_{fact_id}"
    return facts_json

def process_video(video_path: str, facts_dir: str, keyframes_dir: str, key_path: str):
    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    folder_name = os.path.basename(os.path.dirname(video_path))
    clip_folder = os.path.join(keyframes_dir, folder_name, clip_name)
    os.makedirs(clip_folder, exist_ok=True)

    print(f"\n Processing: {video_path}")

    try:
        response = extract_visual_facts_from_video(video_path, key_path)
        frame_ids = get_frame_ids_from_facts(response)

        json_response = smart_json_loads(response)
        facts_json = add_fact_ids_to_facts_json(json_response)

        adjusted_ids_map = {}

        for fact in facts_json.get("facts", []):
            fid = fact.get("timestamp", None)
            if fid is not None:
                fid_seconds = parse_timestamp_to_seconds(str(fid))
                adjusted_seconds = int(clip_name) * 30 + fid_seconds
                adjusted_timestamp = seconds_to_timestamp(adjusted_seconds)
                adjusted_ids_map[fid_seconds] = adjusted_timestamp
                fact["timestamp"] = adjusted_timestamp
            for voice in fact.get("asr_periods", []):
                adjusted_start = int(clip_name) * 30 + parse_timestamp_to_seconds(voice.get("starttime"))
                adjusted_end = int(clip_name) * 30 + parse_timestamp_to_seconds(voice.get("endtime"))
                voice["starttime"] = seconds_to_timestamp(adjusted_start)
                voice["endtime"] = seconds_to_timestamp(adjusted_end)

        keyframes = extract_frames_and_save(video_path, frame_ids, clip_folder, adjusted_ids_map)

        for fact in facts_json.get("facts", []):
            adjusted_ts = fact.get("timestamp")
            if adjusted_ts is not None:
                fact["key_frames"] = keyframes.get(adjusted_ts, [])

        output_json_path = os.path.join(facts_dir, f"{clip_name}.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_response, f, indent=2, ensure_ascii=False)

        print(f"Finished: {clip_name}")
        return clip_name, "ok"

    except Exception as e:
        traceback.print_exc()
        return clip_name, f"error: {e}"


def run_batch_from_txt(list_file: str, out_facts: str, out_keyframes: str, key_path: str, threads: int):
    os.makedirs(out_facts, exist_ok=True)
    os.makedirs(out_keyframes, exist_ok=True)

    with open(list_file, "r", encoding="utf-8") as f:
        video_folders = [x.strip() for x in f.readlines() if x.strip()]

    for folder in video_folders:
        if not os.path.isdir(folder):
            print(f"[!] Skip non-folder path: {folder}")
            continue

        folder_name = os.path.basename(os.path.normpath(folder))
        print(f"\n[📁] Processing folder: {folder_name}")

        mp4_files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp4")],
            key=natural_sort_key
        )

        if not mp4_files:
            print(f"[!] No videos found in {folder}")
            continue

        folder_results = {}

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {
                executor.submit(process_video, v, out_facts, out_keyframes, key_path): v
                for v in mp4_files
            }

            for fut in as_completed(futures):
                v = futures[fut]
                clip_name = os.path.splitext(os.path.basename(v))[0]
                try:
                    clip_name, status = fut.result()

                    video_json_path = os.path.join(out_facts, f"{clip_name}.json")
                    if os.path.exists(video_json_path):
                        with open(video_json_path, "r", encoding="utf-8") as vf:
                            data = json.load(vf)
                        folder_results[clip_name] = data

                        os.remove(video_json_path)
                except Exception as e:
                    print(f"[x] Error processing {v}: {e}")

        output_json_path = os.path.join(out_facts, f"{folder_name}.json")
        sorted_folder_results = dict(sorted(folder_results.items(), key=lambda x: natural_sort_key(x[0])))

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(sorted_folder_results, f, indent=2, ensure_ascii=False)

        print(f"Folder done: {folder_name} → {output_json_path}")


# ================== Entry ==================
def main():
    parser = argparse.ArgumentParser(description="Batch OpenAI Video Facts Extraction (rate limited).")
    parser.add_argument("--list", default="/path/to/video_list.txt", help="Path to txt file (each line = video path)")
    parser.add_argument("--facts", default="./data/facts", help="Output folder for facts JSONs")
    parser.add_argument("--keyframes", default="./data/key_frames", help="Output folder for extracted keyframes")
    parser.add_argument("--key", default="../../config/gemini_key.txt", help="Gemini API key path")
    parser.add_argument("--threads", type=int, default=10, help="Concurrent threads (default: 10)")
    args = parser.parse_args()

    run_batch_from_txt(args.list, args.facts, args.keyframes, args.key, args.threads)


if __name__ == "__main__":
    main()
