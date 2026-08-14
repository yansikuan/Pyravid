import base64
import glob
import json
import os
from typing import Any, Dict, List
import tiktoken
import re, string
import logging
import re
import ast

from pathlib import Path
import networkx as nx

from prototype.tools.api_client import APIClient

encoding = tiktoken.get_encoding('cl100k_base')
logger = logging.getLogger(__name__)

def extract_agentic_action(response: str):
    response = (response or "").strip()

    if not response:
        return {"action": "INVALID"}

    match = re.search(r"\[\s*ANSWER\s*\]\s*([A-D])", response, re.IGNORECASE)
    if match:
        return {
            "action": "ANSWER",
            "answer": match.group(1).upper()
        }

    search_match = re.search(r"\[\s*SEARCH\s*\]\s*(.+)", response, re.IGNORECASE)
    if search_match:
        query = search_match.group(1).strip()
        if query:
            return {
                "action": "SEARCH",
                "query": query
            }
        return {"action": "INVALID"}

    if re.search(r"\[\s*EXPAND\s*\]", response, re.IGNORECASE):
        return {"action": "EXPAND"}

    return {"action": "INVALID"}

def extract_subgoals(response: str) -> List[str]:

    subgoals = []

    # Pattern to match "Sub-goal N: [text]"
    pattern = r'Sub-goal\s+\d+:\s*(.+?)(?=\nSub-goal|\Z)'

    matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)

    for match in matches:
        # Clean up the sub-goal text
        subgoal = match.strip()
        if subgoal:
            subgoals.append(subgoal)

    return subgoals

def extract_ge_agentic_action_v4(response: str) -> Dict:
    """Extract action from AGENTIC_GE_EXPAND_PROMPT_V4 response."""
    import re

    result = {
        "action": "INVALID",
        "satisfied_subgoals": [],
        "reasoning": "",
        "answer": None
    }

    # Extract ACTION
    action_match = re.search(r'ACTION:\s*(EXPAND|ANSWER)', response, re.IGNORECASE)
    if action_match:
        result["action"] = action_match.group(1).upper()

    # Extract SATISFIED_SUBGOALS
    subgoals_match = re.search(r'SATISFIED_SUBGOALS:\s*\[(.*?)\]', response)
    if subgoals_match:
        subgoals_str = subgoals_match.group(1)
        try:
            result["satisfied_subgoals"] = [int(s.strip()) for s in subgoals_str.split(',') if s.strip()]
        except ValueError:
            result["satisfied_subgoals"] = []

    # Extract REASONING
    reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\[ANSWER\]|ACTION:|SATISFIED_SUBGOALS:|\Z)', response, re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract ANSWER (only for ANSWER action)
    if result["action"] == "ANSWER":
        answer_match = re.search(r'\[ANSWER\]\s*(.+)', response, re.DOTALL)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
        else:
            result["action"] = "INVALID"

    return result

def extract_ge_agentic_action(response: str):
    response = (response or "").strip()

    if not response:
        return {"action": "INVALID"}

    answer_match = re.search(
        r"\[\s*ANSWER\s*\]\s*(.+)",
        response,
        re.IGNORECASE | re.DOTALL
    )
    if answer_match:
        answer_text = answer_match.group(1).strip()
        if answer_text:
            return {
                "action": "ANSWER",
                "answer": answer_text
            }
        return {"action": "INVALID"}

    search_match = re.search(
        r"\[\s*SEARCH\s*\]\s*(.+)",
        response,
        re.IGNORECASE
    )
    if search_match:
        query = search_match.group(1).strip()
        if query:
            return {
                "action": "SEARCH",
                "query": query
            }
        return {"action": "INVALID"}

    if re.search(r"\[\s*EXPAND\s*\]", response, re.IGNORECASE):
        return {"action": "EXPAND"}

    return {"action": "INVALID"}

def extract_prediction(response: str):
    match = re.search(r"\[ANSWER\]\s*([ABCD])\b", response)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract prediction from response: {response}")


def extract_answer(model_output: str) -> str:
    """
    Extract the answer content after '[ANSWER]'
    """
    pattern = r"\[ANSWER\]\s*(.*)$"
    match = re.search(pattern, model_output, re.DOTALL)

    answer = match.group(1).strip() if match else ""
    if not answer:
        logger.warning(f"Failed to extract answer from model output: {model_output}")
        raise ValueError(f"Cannot extract answer from model output: {model_output}")
    return answer

def load_memory_graph(graph_path: str | Path) -> nx.Graph:
    G_loaded : nx.Graph = nx.read_gexf(graph_path)
    for node, data in G_loaded.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                G_loaded.nodes[node][key] = ast.literal_eval(value)
    mapping = {node: int(node) for node in G_loaded.nodes()}
    G_loaded = nx.relabel_nodes(G_loaded, mapping)
    return G_loaded

def llm_judge(client: APIClient,
              question: str,
              prediction: str,
              answer: str,
              max_tokens: int = 500,
              temperature: float = 0.0) -> float:
    """
    Use an LLM to judge if `prediction` semantically matches the gold `answer`.
    Returns 1.0 for "Yes", 0.0 otherwise.
    """
    from prototype.tools.prompts import judge_template
    prompt = judge_template.format(question=question, prediction=prediction, answer=answer)
    resp = client.obtain_response(prompt, max_tokens=max_tokens, temperature=temperature).strip()
    return 1.0 if resp.lower() == "yes" else 0.0

def llm_judge_m3(client: APIClient,
              question: str,
              prediction: str,
              answer: str,
              max_tokens: int = 500,
              temperature: float = 0.0) -> float:
    from prototype.tools.prompts import judge_template_m3
    prompt = judge_template_m3.format(question=question, ground_truth_answer=answer, agent_answer=prediction)
    resp = client.obtain_response(prompt, max_tokens=max_tokens, temperature=temperature).strip()
    return 1.0 if resp.lower() == "yes" else 0.0

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def F1(answer, key):
    key = normalize_answer(key)
    answer = normalize_answer(answer)
    f1_score = calculate_f1_score(answer, key)
    return f1_score

def EM(answer, key):
    return normalize_answer(answer) == normalize_answer(key)

def calculate_f1_score(predicted_answer, true_answer):
    predicted_set = set(re.split(r'[ -]', predicted_answer))
    true_set = set(re.split(r'[ -]', true_answer))

    if len(predicted_set) == 0:
        return 0

    true_positives = len(predicted_set.intersection(true_set))

    precision = true_positives / len(predicted_set)
    recall = true_positives / len(true_set)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1

def generate_qwen_messages(input, system_prompt="You are an expert in multimodal understanding."):
    """Generate message for Qwen unimodal model.
    Args:
        input (str): Input user message.
        system_prompt (str): System prompt.
    Returns:
        list: Formatted messages for chat completion.
    """
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": input}]

def generate_qwen_omni_messages(inputs):
    """Generate message list for chat completion from mixed inputs.

    Args:
        inputs (list): List of input dictionaries with 'type' and 'content' keys
        type can be:
            "text" - text content
            "image/jpeg", "image/png" - base64 encoded images
            "video/mp4", "video/webm" - base64 encoded videos
            "video_url" - video URL
            "audio/mp3", "audio/wav" - base64 encoded audio
        content should be a string for text,
        a list of base64 encoded media for images/video/audio,
        or a string (url) for video_url
        inputs are like:
        [
            {
                "type": "video_base64/mp4",
                "content": <base64>
            },
            {
                "type": "text",
                "content": "Describe the video content."
            },
            ...
        ]

    Returns:
        list: Formatted messages for chat completion
    """
    messages = []
    content = []
    for input in inputs:
        if not input["content"]:
            logger.warning("empty content, skip")
            continue
        if input["type"] == "text":
            content.append({"type": "text", "text": input["content"]})
        elif input["type"] in ["images/jpeg", "images/png"]:
            img_format = input["type"].split("/")[1]
            if isinstance(input["content"][0], str):
                content.extend(
                    [
                        {
                            "type": "image",
                            "image": f"data:image;base64,{img}",
                        }
                        for img in input["content"]
                    ]
                )
            else:
                for img in input["content"]:
                    content.append({
                        "type": "text",
                        "text": img[0],
                    })
                    content.append({
                        "type": "image",
                        "image": f"data:image;base64,{img[1]}"
                    })
        elif input["type"] in ["video_url", "video_base64/mp4", "video_base64/webm", "vllm_video_base64/mp4"]:
            content.append(
                {
                    "type": "video",
                    "video": input["content"],
                }
            )
        else:
            raise ValueError(f"Invalid input type: {input['type']}")
    messages.append({"role": "user", "content": content})
    return messages

def generate_messages(inputs, system_prompt="You are an expert in multimodal understanding."):
    """Generate message list for chat completion from mixed inputs.

    Args:
        inputs (list): List of input dictionaries with 'type' and 'content' keys
        system_prompt (str): System message content

    Returns:
        list: Formatted messages for chat completion
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    content = []

    for input_item in inputs:
        if not input_item.get("content"):
            logger.warning("empty content, skip")
            continue

        input_type = input_item.get("type", "")
        input_content = input_item.get("content")

        if input_type == "text":
            content.append({"type": "text", "text": input_content})

        elif input_type in ["images/jpeg", "images/png"]:
            img_format = input_type.split("/")[1]
            if isinstance(input_content, list) and input_content and isinstance(input_content[0], str):
                # list of base64 strings
                content.extend([
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img_format};base64,{img}",
                            "detail": "high",
                        },
                    }
                    for img in input_content
                ])
            else:
                # list of (caption, base64) tuples
                for img in input_content:
                    content.append({"type": "text", "text": img[0]})
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img_format};base64,{img[1]}",
                            "detail": "high",
                        },
                    })

        elif input_type == "video_url":
            content.append({
                "type": "image_url",
                "image_url": {"url": input_content},
            })

        elif input_type in ["video_base64/mp4", "video_base64/webm"]:
            video_format = input_type.split("/")[1]
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:video/{video_format};base64,{input_content}"},
            })
        elif input_type in ["audio_base64/mp3", "audio_base64/wav"]:
            audio_format = input_type.split("/")[1]
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:audio/{audio_format};base64,{input_content}"}
            })
        elif input_type in ["vllm_video_base64/mp4"]:
            video_format = input_type.split("/")[1]
            content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/{video_format};base64,{input_content}"},
            })
        else:
            raise ValueError(f"Invalid input type: {input_type}")

    messages.append({"role": "user", "content": content})
    return messages

def smart_json_loads(text: str):

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    candidates = [text]
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match and match.group(0) != text:
        candidates.append(match.group(0))

    # Character-processing model output occasionally contains literal newlines
    # or typographic quotes. Keep the existing parser as the primary path and
    # use this normalization only after its candidates have failed.
    fallback = (
        text.replace("\n", "")
            .replace("\r", "")
            .replace("“", '"')
            .replace("”", '"')
            .replace("‘", '"')
            .replace("’", '"')
    )
    if fallback != text:
        candidates.append(fallback)
        fallback_match = re.search(r'\{.*\}', fallback, re.DOTALL)
        if fallback_match and fallback_match.group(0) != fallback:
            candidates.append(fallback_match.group(0))

    last_error = None
    for candidate in dict.fromkeys(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_error = e

    if any(re.search(r'\{.*\}', candidate, re.DOTALL) for candidate in candidates):
        raise ValueError(f"Failed to parse JSON: {last_error}")

    raise ValueError("No valid JSON found in the provided text.")


def collect_faces_by_clip(char_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    idx: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.exists(char_dir):
        print(f"[⚠️] Faces folder not found: {char_dir}")
        return idx

    print(f"[🔍] Collecting faces from {char_dir} ...")

    for path in glob.glob(os.path.join(char_dir, "**/*.json"), recursive=True):
        try:
            data = load_facts(path)
        except Exception as e:
            print(f"[⚠️] Failed to read {path}: {e}")
            continue

        face_id = data.get("face_id") or os.path.splitext(os.path.basename(path))[0]

        for face in data.get("faces", []):
            clip = str(face.get("clip_name", "0"))
            fact_id = str(face.get("fact_id", "unknown_fact"))
            image_base64 = face.get("image_base64")

            if not image_base64 and face.get("image_path"):
                image_path = face["image_path"]
                if os.path.exists(image_path):
                    try:
                        with open(image_path, "rb") as image_file:
                            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
                    except Exception as e:
                        print(f"[⚠️] Failed to read image: {image_path} ({e})")
                else:
                    print(f"[⚠️] Image path not found: {image_path}")

            if not image_base64:
                continue

            idx.setdefault(clip, []).append({
                "face_id": face_id,
                "image_base64": image_base64,
                "fact_id": fact_id,
                "frame_id": face.get("frame_id"),
            })

    print(f"[✅] Faces collected for {len(idx)} clips: {list(idx.keys())}")
    return idx


def collect_voices_by_clip(voices_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Collect all voices from a single clip directory."""
    idx: Dict[str, List[Dict[str, Any]]] = {}

    if not os.path.exists(voices_dir):
        print(f"[⚠️] Voices folder not found: {voices_dir}")
        return idx

    clip_name = os.path.basename(voices_dir.rstrip("/"))
    idx[clip_name] = []

    print(f"[🔍] Collecting voices from {voices_dir} ...")

    for path in glob.glob(os.path.join(voices_dir, "*.json")):
        try:
            data = load_facts(path)
        except Exception as e:
            print(f"[⚠️] Failed to read {path}: {e}")
            continue

        segments = [data] if isinstance(data, dict) else (data if isinstance(data, list) else [])

        for segment in segments:
            audio_base64 = segment.get("audio_base64", "")
            if isinstance(audio_base64, str) and audio_base64:
                if audio_base64.startswith("data:audio"):
                    audio_base64 = audio_base64.split("base64,", 1)[-1]
                audio_base64 = (
                    audio_base64.strip().replace("\n", "").replace("\r", "").replace(" ", "")
                )
            else:
                audio_base64 = ""

            idx[clip_name].append({
                "voice_id": segment.get("voice_id") or segment.get("voiceId") or segment.get("id"),
                "fact_id": segment.get("fact_id"),
                "asr": segment.get("asr_text") or segment.get("asr") or "",
                "audio_base64": audio_base64,
                "start_time": segment.get("start_time", 0),
                "end_time": segment.get("end_time", 0),
            })

    print(f"[✅] Voices collected: {len(idx[clip_name])} segments.")
    return idx


def call_model(
    inputs: list,
    system_prompt: str,
    model: str = "gemini-2.5-flash",
    max_tokens: int = 8192,
) -> str:
    """Call a Gemini multimodal model for character-processing reasoning."""
    key_path = "./config/gemini_key.txt"
    try:
        client = APIClient(
            api="gemini",
            key_path=key_path,
            model=model,
            embedding_model="",
        )
        messages = generate_messages(inputs, system_prompt=system_prompt)
        print(f"[🧠] Sending {len(inputs)} multimodal inputs to model...")
        response = client.obtain_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return response or ""
    except Exception as e:
        print(f"[⚠️] Model call failed: {e}")
        return ""


def load_facts(facts_path: str) -> dict:
    with open(facts_path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_timestamp_to_seconds(timestamp: str) -> float:
    timestamp = timestamp.strip()
    parts = timestamp.split(":")
    try:
        if len(parts) >= 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
            return minutes * 60 + seconds
        if len(parts) == 1:
            return float(parts[0])
    except (ValueError, IndexError):
        pass

    try:
        return float(timestamp)
    except ValueError:
        print(f"[!] Failed to parse timestamp: {timestamp}")
        return 0.0


def extract_keyframes_from_fact(fact: dict):
    frames = []
    timestamp = fact.get("timestamp")
    print(f"Extracting keyframes for fact_id: {fact.get('id')} with timestamp: {timestamp}")
    for frame_info in fact.get("key_frames", []):
        b64_path = frame_info.get("b64_path")
        with open(b64_path, "r") as file:
            base64_frame = file.read().strip()
        base64_frame = base64_frame.split(",")[1]
        frames.append({
            "fact_id": fact.get("id"),
            "timestamp": timestamp,
            "b64_path": b64_path,
            "base64_frame": base64_frame,
            "jpg_path": frame_info.get("jpg_path"),
        })
    return frames


def seconds_to_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def save_json(data: Any, path: str, person_file: bool = False):
    existing_data = load_facts(path) if os.path.exists(path) else {}

    if person_file:
        new_facts = data.get("facts", [])
        existing_facts = existing_data.get("facts", [])

        for new_fact in new_facts:
            if not any(fact.get("fact_id") == new_fact.get("fact_id") for fact in existing_facts):
                existing_facts.append(new_fact)

        existing_data["facts"] = existing_facts
        existing_voice_ids = set(existing_data.get("voice_ids", []))
        new_voice_ids = set(data.get("voice_ids", []))
        existing_data["voice_ids"] = list(existing_voice_ids | new_voice_ids)

        if data.get("face_id"):
            existing_data["face_id"] = data.get("face_id")

        for key in data:
            if key not in ["facts", "voice_ids", "face_id"]:
                existing_data[key] = data[key]

        data = existing_data
    else:
        existing_data.update(data)
        data = existing_data

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
