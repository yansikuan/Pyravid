"""
Socratic Memory baseline. Modes:
- convert: re-encode all clips to H.264: <clips_root>/<video_id>/*.mp4 → <output_clips_root>/<video_id>/*.mp4
- ingest: video list from question_dir; for each, extract memory from <clips_root>/<video_id>/*.mp4 → <vector_store_path>/<video_id>
- answer: load memory from vector store, run MC QA (lvbench format), write predictions to save_dir.
Both ingest and answer use question_dir (or ./data/<dataset>/questions/) to determine which videos to process.
"""

from __future__ import annotations

import subprocess
import tempfile
import argparse
import base64
import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

from tqdm import tqdm

from prototype.tools.api_client import APIClient
from prototype.tools.vectorstore import Qdrant
from prototype.tools.utils import generate_messages, llm_judge_m3, generate_qwen_omni_messages

# ---------- Prompts ----------
CLIP_MEMORY_PROMPT = (
    "Write one concise paragraph describing what happens: main actions, objects, people, "
    "setting, and any spoken or on-screen text. This will be used later to answer questions "
    "about the video. Output only the description, no JSON or labels."
)

SOCRATIC_MC_PROMPT = """Answer the question based only on the following context. If the answer is not in the context, give your best guess.

**Context:**
{context}

**Question:** {question}

**Options:**
{options}

**Your Answer (A, B, C, or D):** """

SOCRATIC_OPEN_PROMPT = """Answer the question based only on the following retrieved context from the video. Be concise. If the context does not contain enough information, give your best guess.

**Context:**
{context}

**Question:** {question}

**Answer:**"""


# ---------- Helpers ----------
def encode_video_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def create_answer_backend_client(args) -> APIClient:
    """Create the model client used for clip description (ingest) and MC answer (answer mode)."""
    if getattr(args, "answer_api", "gemini") == "qwen-server":
        return APIClient(
            "qwen-server-answer",
            None,
            args.answer_model,
            None,
            ip_address=getattr(args, "answer_api_ip", "localhost"),
        )
    if getattr(args, "answer_api", "gemini") == "qwen-omni":
        return APIClient("qwen-omni", None, args.answer_model, None)
    return APIClient("gemini", args.answer_api_key_path, args.answer_model, None)


# ---------- Memory generation (ingestion) ----------
def process_clip_to_memory(
    clip_path: str,
    clip_id: int,
    video_id: str,
    vision_client: APIClient,
    embedding_client: APIClient,
    qdrant_store: Qdrant,
    point_id: Optional[Any] = None,
    answer_api: str = "gemini",
) -> Optional[str]:
    """Process one clip to memory. Returns description or None if clip is skipped (problematic codec or server error)."""
    try:
        video_b64 = encode_video_b64(clip_path)
        if answer_api != "qwen-omni":
            inputs = [
                {"type": "text", "content": CLIP_MEMORY_PROMPT},
                {"type": "video_base64/mp4", "content": video_b64} if answer_api == "gemini" else {"type": "vllm_video_base64/mp4", "content": video_b64},
            ]
        else:
            inputs = [
                {"type": "text", "content": CLIP_MEMORY_PROMPT},
                {"type": "video_base64/mp4", "content": video_b64},
            ]
        if answer_api != "qwen-omni":
            messages = generate_messages(inputs, system_prompt="You are a video summarizer for long-term memory.")
        else:
            messages = generate_qwen_omni_messages(inputs)
            messages.append({"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]})

        description = vision_client.obtain_response(
            messages=messages, max_tokens=1000, temperature=0.0
        ).strip()
        embedding = embedding_client.obtain_embedding(description)
        vector = list(embedding) if not isinstance(embedding, list) else embedding
        payload = {"clip_id": clip_id, "description": description, "video_id": video_id}
        id_seed = point_id if point_id is not None else f"{video_id}_{clip_id}"
        id_ = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(id_seed)))
        qdrant_store.insert(vectors=[vector], payloads=[payload], ids=[id_])
        return description
    except Exception as e:
        logging.warning("[Skip] Failed to process clip, skipping: %s — %s", clip_path, e)
        return None


def ingest_clip_folder(
    clips_folder: str,
    video_id: str,
    vision_client: APIClient,
    embedding_client: APIClient,
    qdrant_store: Qdrant,
    extension: str = ".mp4",
    answer_api: str = "gemini",
) -> List[str]:
    """Extract memory from all MP4s in clips_folder; store in qdrant_store."""
    files = [f for f in os.listdir(clips_folder) if f.lower().endswith(extension)]
    files.sort(key=natural_sort_key)
    descriptions = []
    for clip_id, filename in tqdm(enumerate(files), total=len(files), desc=f"Processing {video_id}", leave=False):
        clip_path = os.path.join(clips_folder, filename)
        desc = process_clip_to_memory(
            clip_path=clip_path,
            clip_id=clip_id,
            video_id=video_id,
            vision_client=vision_client,
            embedding_client=embedding_client,
            qdrant_store=qdrant_store,
            answer_api=answer_api,
        )
        if desc is not None:
            descriptions.append(desc)
    return descriptions

def _ingest_one_video(
    video_id: str,
    clips_folder: str,
    path_for_video: str,
    args: argparse.Namespace,
    embedding_client: APIClient,
    vision_client: APIClient,
) -> str:
    """Ingest one video: create store, run ingest_clip_folder, close. Returns video_id."""
    os.makedirs(path_for_video, exist_ok=True)
    qdrant_store = Qdrant(
        collection_name=video_id,
        embedding_model_dims=args.embedding_dims,
        path=path_for_video,
        on_disk=True,
    )
    try:
        ingest_clip_folder(
            clips_folder=clips_folder,
            video_id=video_id,
            vision_client=vision_client,
            embedding_client=embedding_client,
            qdrant_store=qdrant_store,
            answer_api=args.answer_api,
        )
    finally:
        qdrant_store.close()
    return video_id

def run_ingest(args):
    """Memory generation only: video list from question_dir; for each, ingest <clips_root>/<video_id>/*.mp4 → vector_store_path. Optional multithreading over videos."""
    question_path = args.question_dir or f"./data/{args.dataset}/questions/"
    if not os.path.isdir(question_path):
        logging.error("--question_dir (or ./data/<dataset>/questions/) must exist for mode=ingest")
        return
    if not args.clips_root or not os.path.isdir(args.clips_root):
        logging.error("--clips_root must be an existing directory for mode=ingest")
        return

    os.makedirs(args.vector_store_path, exist_ok=True)
    embedding_client = APIClient("openai", args.api_key_path, args.model, args.embedding_model)
    vision_client = create_answer_backend_client(args)

    files = [f for f in os.listdir(question_path) if f.endswith(".json")]
    files.sort()
    video_ids = [os.path.splitext(f)[0] for f in files]

    # Build list of (video_id, clips_folder, path_for_video) that need processing
    tasks = []
    for video_id in video_ids:
        clips_folder = os.path.join(args.clips_root, video_id)
        path_for_video = os.path.join(args.vector_store_path, video_id)
        if os.path.isdir(path_for_video):
            logging.info("[Skip] Memory already present for %s at %s", video_id, path_for_video)
            continue
        if not os.path.isdir(clips_folder):
            logging.warning("[Skip] No clips folder for %s at %s", video_id, clips_folder)
            continue
        mp4_files = [f for f in os.listdir(clips_folder) if f.lower().endswith(".mp4")]
        if not mp4_files:
            logging.warning("[Skip] No MP4s in %s", clips_folder)
            continue
        tasks.append((video_id, clips_folder, path_for_video))

    num_workers = getattr(args, "num_workers", 1)

    if num_workers <= 1:
        for video_id, clips_folder, path_for_video in tqdm(tasks, desc="Videos"):
            _ingest_one_video(video_id, clips_folder, path_for_video, args, embedding_client, vision_client)
            logging.info("[Ingest] Stored memory for %s", video_id)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _ingest_one_video,
                    video_id,
                    clips_folder,
                    path_for_video,
                    args,
                    embedding_client,
                    vision_client,
                ): video_id
                for (video_id, clips_folder, path_for_video) in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Videos"):
                vid = futures[future]
                try:
                    future.result()
                    logging.info("[Ingest] Stored memory for %s", vid)
                except Exception as e:
                    logging.exception("[Ingest] Failed for %s: %s", vid, e)

    print("[Ingest] Done.")


# ---------- Answer (MC QA) ----------
def answer_question_mc(
    question: str,
    options: List[str],
    embedding_client: APIClient,
    answer_client: APIClient,
    qdrant_store: Qdrant,
    video_id: Optional[str] = None,
    top_k: int = 10,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    q_emb = embedding_client.obtain_embedding(question)
    query_vector = list(q_emb) if not isinstance(q_emb, list) else q_emb
    filters = None
    if video_id is not None:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            filters = Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))])
        except Exception:
            pass
    hits = qdrant_store.search(
        query=None,
        vectors=query_vector,
        limit=top_k,
        filters=filters,
    )
    context_parts = []
    for h in hits:
        p = h.payload if hasattr(h, "payload") else h
        if isinstance(p, dict):
            context_parts.append(p.get("description", ""))
    context = "\n\n".join(context_parts) if context_parts else "No relevant segments found."
    opts_text = "\n".join(options)
    prompt = SOCRATIC_MC_PROMPT.format(
        context=context,
        question=question,
        options=opts_text,
    )
    messages = generate_messages([{"type": "text", "content": prompt}], system_prompt="You are an assistant that answers multiple-choice questions using only the given context. Output exactly one letter: A, B, C, or D. No explanation or extra text.")
    response = answer_client.obtain_response(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.strip()


def answer_question_open(
    question: str,
    embedding_client: APIClient,
    answer_client: APIClient,
    qdrant_store: Qdrant,
    video_id: Optional[str] = None,
    top_k: int = 10,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Retrieve relevant clip descriptions and generate a free-form answer."""
    q_emb = embedding_client.obtain_embedding(question)
    query_vector = list(q_emb) if not isinstance(q_emb, list) else q_emb
    hits = qdrant_store.search(
        query=None,
        vectors=query_vector,
        limit=top_k,
    )
    context_parts = []
    for h in hits:
        p = h.payload if hasattr(h, "payload") else h
        if isinstance(p, dict):
            context_parts.append(p.get("description", ""))
    context = "\n\n".join(context_parts) if context_parts else "No relevant segments found."
    prompt = SOCRATIC_OPEN_PROMPT.format(context=context, question=question)
    messages = [{"role": "user", "content": prompt}]
    response = answer_client.obtain_response(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.strip()


def run_answer(args):
    """Answer only: load existing vector store per video, run MC QA, save predictions."""
    question_path = args.question_dir or f"./data/{args.dataset}/questions/"
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, args.save_name or "socratic_qa_output.json")

    prediction_json = []
    total_num = 0
    correct_num = 0

    embedding_client = APIClient("openai", args.api_key_path, args.model, args.embedding_model)
    answer_client = create_answer_backend_client(args)

    files = [f for f in os.listdir(question_path) if f.endswith(".json")]
    files.sort()

    for file_name in files:
        video_id = os.path.splitext(file_name)[0]
        path_for_video = os.path.join(args.vector_store_path, video_id)
        if not os.path.isdir(path_for_video):
            logging.warning("[Skip] No vector store for %s at %s", video_id, path_for_video)
            continue
        print(f"\n=== Answering Questions in {video_id} ===")
        qdrant_store = Qdrant(
            collection_name=video_id,
            embedding_model_dims=args.embedding_dims,
            path=path_for_video,
            on_disk=True,
        )

        with open(os.path.join(question_path, file_name), "r") as f:
            question_list = json.load(f)

        total_num += len(question_list)
        local_correct = 0

        for qd in tqdm(question_list, leave=False):
            qid = qd.get("QID")
            question = qd["Question"]
            options = qd.get("Options")
            gold = qd.get("Gold")
            answer = qd.get("Answer")

            if options is not None and gold is not None:
                prediction = answer_question_mc(
                    question=question,
                    options=options,
                    embedding_client=embedding_client,
                    answer_client=answer_client,
                    qdrant_store=qdrant_store,
                    video_id=video_id,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                is_correct = (prediction.upper() == str(gold).strip().upper())
                local_correct += 1 if is_correct else 0
                rec = {
                    "QID": qid,
                    "Question": question,
                    "Options": options,
                    "Gold": gold,
                    "Prediction": prediction,
                    "Correct": is_correct,
                }
                print(f"Question: {question}")
                print(f"Prediction: {prediction}")
                print(f"Gold: {gold}")
            else:
                prediction = answer_question_open(
                    question=question,
                    embedding_client=embedding_client,
                    answer_client=answer_client,
                    qdrant_store=qdrant_store,
                    video_id=video_id,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                rec = {"QID": qid, "Question": question, "Answer": answer, "Prediction": prediction}
                print(f"Question: {question}")
                print(f"Prediction: {prediction}")
                print(f"Answer: {answer}")
                if answer is not None:
                    score_llm = llm_judge_m3(
                        client=answer_client,
                        question=question,
                        prediction=prediction,
                        answer=answer,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    rec["LLM_Judge"] = score_llm
                    local_correct += score_llm
                print(f"LLM Judge: {score_llm}")
            prediction_json.append(rec)

        acc = local_correct / max(1, len(question_list))
        correct_num += local_correct
        print(f"[{video_id}] Accuracy: {acc:.4f}")
        print("=" * 25)
        qdrant_store.close()

    if total_num == 0:
        print("[Warn] No questions processed.")
    else:
        print(f"\n== Final accuracy: {correct_num / total_num:.4f} ==")

    with open(out_path, "w") as f:
        json.dump(prediction_json, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_path}")


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Socratic baseline: memory generation (ingest) or answer (MC QA). Both use question_dir to determine which videos.")
    p.add_argument("--mode", type=str, choices=["convert", "ingest", "answer"], default="ingest",
                   help="convert = re-encode all clips to H.264; ingest = build memory from clips; answer = run MC QA using existing memory")
    p.add_argument("--dataset", type=str, default="videomme-test")
    p.add_argument("--question_dir", type=str, default=None,
                   help="Directory of per-video question JSONs (for convert/ingest/answer). Default: ./data/<dataset>/questions/")
    p.add_argument("--clips_root", type=str, default="/path/to/external/m3-agent-videomme/data/clips",
                   help="Root for clips per video: <clips_root>/<video_id>/*.mp4 (for mode=convert and mode=ingest)")
    p.add_argument("--vector_store_path", type=str, default="./vectorstore_test/socratic/videomme-test/gemini_2.0-flash/qdrant")
    p.add_argument("--api_key_path", type=str, default="./config/openai_key.txt")
    p.add_argument("--answer_api_key_path", type=str, default="./config/gemini_key.txt")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--embedding_model", type=str, default="text-embedding-3-large")
    p.add_argument("--embedding_dims", type=int, default=3072)
    p.add_argument("--answer_api", type=str, choices=["gemini", "qwen-server", "qwen-omni"], default="gemini", help="Backend for answer (and for ingest: clip description). Default: gemini.")
    p.add_argument("--answer_model", type=str, default="gemini-2.0-flash", help="Model for answer. Default: gemini-2.0-flash.")
    p.add_argument("--answer_api_address", type=str, default="localhost", help="Host for Qwen server (when --answer_api=qwen-server). Default: localhost.")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=10000)
    p.add_argument("--save_dir", type=str, default="./output/baselines/socratic")
    p.add_argument("--save_name", type=str, default=None)
    p.add_argument("--save_evidence", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=1, help="Number of workers for ingest. Default: 1.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.mode == "ingest":
        run_ingest(args)
    else:
        run_answer(args)


if __name__ == "__main__":
    main()