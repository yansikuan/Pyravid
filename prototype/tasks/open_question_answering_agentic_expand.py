import logging
import os
import argparse
import json
import re
import uuid
import ast
import time
import csv

from typing_extensions import override
from typing import Any, Dict, List, Optional, Set, Tuple
from statistics import mean

import numpy as np

from tqdm import tqdm
from prototype.tools.prompts import agentic_ge_expand_template_v2_without_reasoning, passage_selection_rerank_template, passage_selection_v3_without_summary_template, passage_selection_v3_template, final_response_ge_multimodal_template, agentic_ge_expand_template_v2, generate_ge_new_query_template, agentic_expand_template_v3, final_response_ge_multimodal_template_v2
from prototype.tools.utils import generate_messages, llm_judge_m3, load_memory_graph, extract_ge_agentic_action, extract_prediction, extract_answer
from prototype.tools.vectorstore import Qdrant
from prototype.tasks.explorer import Explorer

MAX_IMAGES_PER_PROMPT = 50
SELECTION_CANDIDATES_THRESHOLD = 50

def _percentile(sorted_values, percentile):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)

class OpenQuestionExplorer(Explorer):
    """
    Prune-and-Grow over the memory hierarchy.
    """
    def __init__(
        self,
        dataset: str,
        video_id: str,
        before_clip_id: int,
        super_graph_dir: str,
        super_embedding_dir: str,
        vector_store_dir: str,
        api_key_path: str = "openai_key.txt",
        answer_api_key_path: str = "gemini_key.txt",
        model: str = "gpt-4o-mini",
        answer_model: str = "gemini-2.0-flash",
        multimodal: bool = True,
        embedding_model: str = "text-embedding-3-large",
        top_k: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 500,
        two_level_mode: bool = True,
        selection_model: str = "gemini-2.0-flash",
        with_top_summary: bool = True,
        data="",
        rerank_with_selection: bool = True,
        with_reasoning: bool = True,
        with_prune: bool = True,
        answer_ip_address: str = "localhost",
        selection_ip_address: str = "localhost"
    ):
        super().__init__(dataset=dataset, video_id=video_id, super_graph_dir=super_graph_dir, super_embedding_dir=super_embedding_dir, api_key_path=api_key_path, answer_api_key_path=answer_api_key_path, model=model, answer_model=answer_model, multimodal=multimodal, embedding_model=embedding_model, top_k=top_k, temperature=temperature, max_tokens=max_tokens, two_level_mode=two_level_mode, selection_model=selection_model, answer_ip_address=answer_ip_address, selection_ip_address=selection_ip_address)
        # Introduce before clip id criteria
        if data == "web":
            self.graph_path = f"./{super_graph_dir}/{dataset}/{video_id}/{video_id}_graph_level_all.gexf"
            self.embedding_path = f"./{super_embedding_dir}/{dataset}/{video_id}/{video_id}_embedding_level_all.npy"
            facts_path = f"{os.environ.get('PYRAVID_FACTS_DIR')}/{video_id}.json"
            with open(facts_path, "r", encoding="utf-8") as f:
              facts_data = json.load(f)
            max_clip = max(int(k) for k in facts_data.keys())
            db = f"./{vector_store_dir}/snapshot/{video_id}/clip_{max_clip}_snapshot"
            self.vector_store_path = db
        elif data == "robot": 
            self.graph_path = f"./{super_graph_dir}/{dataset}/{video_id}/{video_id}_graph_level_all_{before_clip_id+1}.gexf"
            self.embedding_path = f"./{super_embedding_dir}/{dataset}/{video_id}/{video_id}_embedding_level_all_{before_clip_id+1}.npy"
            self.vector_store_path = f"./{vector_store_dir}/snapshot/{video_id}/clip_{before_clip_id+1}_snapshot"
            self.graph_path = self.graph_path if os.path.exists(self.graph_path) else f"./{super_graph_dir}/{dataset}/{video_id}/{video_id}_graph_level_all.gexf"
            self.embedding_path = self.embedding_path if os.path.exists(self.embedding_path) else f"./{super_embedding_dir}/{dataset}/{video_id}/{video_id}_embedding_level_all.npy"
            self.vector_store_path = self.vector_store_path if os.path.exists(self.vector_store_path) else f"./{vector_store_dir}/{video_id}/snapshot/clip_{before_clip_id}_snapshot"
        #Reload the memory graph and embedding
        self.memory_graph = load_memory_graph(self.graph_path)
        self.memory_embedding = np.load(self.embedding_path).astype(np.float32)
        if self.two_level_mode:
            self.memory_embedding = self.memory_embedding[:-1, :]
        # Initialize the character profile database
        self.character_profile_database = Qdrant(collection_name="character", embedding_model_dims=self.memory_embedding.shape[1], path=self.vector_store_path, on_disk=True)
        self.with_top_summary = with_top_summary
        self.rerank_with_selection = rerank_with_selection
        self.with_reasoning = with_reasoning
        self.with_prune = with_prune

    def _extract_related_character_profiles(self, node_set: Set[Tuple]):
        pattern = r'<([^<>]*_[^<>]*)>' # Extract the character name from the text
        character_matches = set()
        for (_, _, txt, _, _,_) in node_set:
            matches = re.findall(pattern, txt)
            character_matches.update(matches)
        character_person_id2uuid = {character_person_id: str(uuid.uuid5(uuid.NAMESPACE_DNS, character_person_id)) for character_person_id in character_matches if character_person_id.split("_")[0] == "person"}
        character_profiles = {}
        for person_id, person_uuid in character_person_id2uuid.items():
            hit = self.character_profile_database.search(query=person_uuid, vectors=None)
            if hit:
                character_profiles[person_id] = hit[0].payload.get("character_profile")
            else:
                raise ValueError(f"No character profile found for {person_id}")
        return character_profiles

    @override
    def _node_text(self, node_id: int) -> str:
        n = int(node_id)
         #txt = self.memory_graph.nodes[n]["character_level_facts"] if "character_level_facts" in self.memory_graph.nodes[n] else self.memory_graph.nodes[n]["text"]
        if n not in self.memory_graph.nodes:
            print(f"[Warning] Node {n} not in memory_graph, skip.")
            return ""
        node_data = self.memory_graph.nodes[n]
        if "character_level_facts" in node_data and node_data["character_level_facts"]:
            txt = node_data["character_level_facts"]
        elif "raw_fact_text" in node_data and node_data["raw_fact_text"]:
            txt = node_data["raw_fact_text"]
        elif "character_level_clip_summary" in node_data and node_data["character_level_clip_summary"]:
            txt = node_data.get("character_level_clip_summary", "")
        else:
            txt = node_data.get("text", "")
        return txt.strip() if isinstance(txt, str) else ""

    @override
    def _node_scene(self, node_id: int) -> str:
        n = int(node_id)
        # Handle both clip level and fact level nodes
        if n not in self.memory_graph.nodes:
            print(f"[Warning] Node {n} not in memory_graph, skip.")
            return ""
        txt = self.memory_graph.nodes[n]["scene_fact_description"] if "scene_fact_description" in self.memory_graph.nodes[n] else self.memory_graph.nodes[n]["scene_clip_summary"]
        return txt.strip() if isinstance(txt, str) else ""

    @override
    def _llm_selection(self, question: str, context_summary: str, character_profiles: Dict[str, str], candidates: List[Tuple]) -> Optional[List[int]]:
        if not candidates:
            return []
        # Passage index starts from 1
        input_candidates = {idx+1: {"text": text, **timestamp} for idx, (_, timestamp, text, asr_periods, _, _) in enumerate(candidates)}
        prompt = passage_selection_v3_template.format(question=question, context_summary=context_summary, character_profiles=json.dumps(character_profiles), passages=json.dumps(input_candidates))
        prompt = generate_messages([{"type": "text", "content": prompt}], system_prompt="You are an expert video comprehension assistant skilled at identifying which text facts/descriptions are useful for answering a given question.")
        resp = self.selection_client.obtain_response(messages=prompt, max_tokens=8000, temperature=self.temperature).strip()
        # logging.info(f"[Selection] Raw response: {resp}")
        #print("LLM Selection Response:", resp)
        try:
            idx_list = ast.literal_eval(resp)
            out = set()
            for i in idx_list:
                j = int(i) - 1
                if 0 <= j < len(candidates):
                    out.add(j)
            return sorted(list(out))
        except Exception:
            return None

    def _llm_rerank(self, question: str, context_summary: str, character_profiles: Dict[str, str], candidates: List[Tuple], top_k: int) -> Optional[List[int]]:

        if not candidates:
            return []
        # Passage index starts from 1
        input_candidates = {idx+1: {"text": text, **timestamp} for idx, (_, timestamp, text, asr_periods, _, _) in enumerate(candidates)}
        prompt = passage_selection_rerank_template.format(question=question, context_summary=context_summary, character_profiles=json.dumps(character_profiles), passages=json.dumps(input_candidates), top_k=top_k)
        prompt = generate_messages([{"type": "text", "content": prompt}], system_prompt="You are an expert video comprehension assistant skilled at identifying which text facts/descriptions are useful for answering a given question.")
        resp = self.selection_client.obtain_response(messages=prompt, max_tokens=8000, temperature=self.temperature).strip()
        # logging.info(f"[Selection] Raw response: {resp}")
        #print("LLM Selection Response:", resp)
        try:
            idx_list = ast.literal_eval(resp)
            out = set()
            for i in idx_list:
                j = int(i) - 1
                if 0 <= j < len(candidates):
                    out.add(j)
            return sorted(list(out))
        except Exception:
            return None


    @override
    def run(self,
            question: str,
            max_exploration_turns: int = 5,
            tolerance: int = 2) -> Tuple[str, List[int], List[Tuple[int, str]]]:
        """
        Returns:
          - prediction (str)
          - used_passages (List[Tuple[int, str]])
        """

        self.answer_client.reset_token_usage()
        self.selection_client.reset_token_usage()
        actual_turns = 0
        total_iter_input_tokens = 0
        total_iter_output_tokens = 0
        final_answer_input_tokens = 0
        final_answer_output_tokens = 0
        per_turn_input_tokens = []
        per_turn_output_tokens = []

        context_summary = self.memory_graph.nodes[list(self.memory_graph.nodes())[-1]].get("text", "")
        print("=== Initial Node Selection ===")
        # Initial top-k
        top_nodes = self._initial_topk(self._embed_question(question))

        candidate_pool = [
        (nid,
         self._node_timestamp(nid),
         self._node_text(nid),
         self._node_asr_periods(nid),
         self._node_image(nid),
         self._node_scene(nid))
        for nid in top_nodes
        ]

        init_retrieved = candidate_pool.copy()

        candidate_node_ids = set(nid for nid, *_ in candidate_pool)
        character_candidate = self._extract_related_character_profiles(candidate_pool)
        all_retrieved = []
        fact_node_images = set()
        combined_evidence = {}
        prediction = None

        for turn in range(max_exploration_turns):
            answer_in_t  = self.answer_client.get_token_usage()["prompt_tokens"]
            answer_out_t = self.answer_client.get_token_usage()["completion_tokens"]
            sel_in_t     = self.selection_client.get_token_usage()["prompt_tokens"]
            sel_out_t    = self.selection_client.get_token_usage()["completion_tokens"]
            local_tol = tolerance
            print(f"=== Exploration Turn {turn} ===")

            if self.with_prune:
                if len(candidate_node_ids) > SELECTION_CANDIDATES_THRESHOLD:
                    print(f"[Select] node_content ({len(candidate_node_ids)}) exceeds threshold ({SELECTION_CANDIDATES_THRESHOLD}), splitting into batches...")
                    all_chosen_indices = set()
                    batch_size = SELECTION_CANDIDATES_THRESHOLD
                    for batch_start in range(0, len(candidate_pool), batch_size):
                        batch_end = min(batch_start + batch_size, len(candidate_pool))
                        batch = candidate_pool[batch_start:batch_end]
                        # Retry logic for each batch
                        batch_chosen_local_idx = None
                        batch_tol = tolerance
                        while batch_chosen_local_idx is None and batch_tol > 0:
                            batch_chosen_local_idx = self._llm_selection(question, context_summary, character_candidate, batch)
                            if batch_chosen_local_idx is None:
                                batch_tol -= 1
                                if batch_tol > 0:
                                    print(f"[Select] parse failed, retrying... ({batch_tol} attempts left)")
                                else:
                                    print("[Select] tolerance exhausted, skipping batch...")
                        if batch_chosen_local_idx is not None:
                            global_indices = {batch_start + idx for idx in batch_chosen_local_idx}
                            all_chosen_indices.update(global_indices)
                    chosen_local_idx = sorted(list(all_chosen_indices)) if all_chosen_indices else []
                # If the number of candidates is less than the threshold, process all candidates at once
                else:
                    chosen_local_idx = self._llm_selection(question, context_summary, character_candidate, candidate_pool)
                if chosen_local_idx is None or len(chosen_local_idx) == 0:
                    if chosen_local_idx is None:
                        print("[Select] parse failed, retrying...")
                        if local_tol > 0:
                            local_tol -= 1
                            continue
                        else:
                            print("[Select] tolerance exhausted, stop.")
                            break
                    else:
                        print("[Select] empty selection.")
                        continue
                chosen_nodes= [candidate_pool[i] for i in chosen_local_idx]
                print(f"LLM selected {len(chosen_nodes)} candidates: {[nid for nid, *_ in chosen_nodes][:30]}{' ...' if len(chosen_nodes) > 30 else ''}")

            else:
                chosen_nodes = candidate_pool.copy()
                print(f"[No Prune] Using all {len(chosen_nodes)} candidates")

            if self.rerank_with_selection:
                top_chosen_local_idx = self._llm_rerank(question, context_summary, character_candidate, chosen_nodes, 10)
                chosen_nodes = [chosen_nodes[i] for i in top_chosen_local_idx]
                print(f"After reranking, top {len(chosen_nodes)} candidates: {[nid for nid, *_ in chosen_nodes][:30]}{' ...' if len(chosen_nodes) > 30 else ''}")

            for nid, timestamp, txt, asr, img, scene in chosen_nodes:
                all_retrieved.append((nid, timestamp, txt, asr, img, scene))
                if self._node_level(nid) == 0 and len(fact_node_images) < MAX_IMAGES_PER_PROMPT:
                    fact_node_images.add(nid)

            all_character_candidate = self._extract_related_character_profiles(all_retrieved)

            combined_evidence = {
            nid: ({"text": txt, **timestamp, "asr_periods": asr, "scene": scene}, img)
            for (nid, timestamp, txt, asr, img, scene) in all_retrieved
            }

            merged_passages, merged_images = self._merge_level0_chunks(
                list(combined_evidence.items()),
                fact_node_images
            )
            if self.with_reasoning:
                prompt = agentic_ge_expand_template_v2.format(
                    question=question,
                    context_summary=context_summary,
                    passages=json.dumps(merged_passages),
                    character_profiles=json.dumps(all_character_candidate)
                )
            else:
                prompt = agentic_ge_expand_template_v2_without_reasoning.format(
                    question=question,
                    context_summary=context_summary,
                    passages=json.dumps(merged_passages),
                    character_profiles=json.dumps(all_character_candidate)
                )

            image_context = {'type': 'images/jpeg', 'content': merged_images}
            text_context = {'type': 'text', 'content': prompt}
            prompt = generate_messages([text_context, image_context],
                                       system_prompt="You are an expert multimodal comprehension assistant skilled at answering questions based on video content.")

            decision_tol = tolerance
            final_chosen_nodes = combined_evidence.keys()

            while decision_tol > 0:

                response = self.answer_client.obtain_response(
                    messages=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                ).strip()
                print(f"LLM response: {response}")

                decision = extract_ge_agentic_action(response)

                if decision["action"] == "INVALID":
                    print(f"[Agent] Invalid response format, retrying... ({decision_tol-1} attempts left)")
                    decision_tol -= 1
                    continue

                elif decision["action"] == "ANSWER":
                    prediction = decision["answer"]
                    actual_turns += 1
                    final_answer_input_tokens = (
                        self.answer_client.get_token_usage()["prompt_tokens"] - answer_in_t +
                        self.selection_client.get_token_usage()["prompt_tokens"] - sel_in_t
                    )
                    final_answer_output_tokens = (
                        self.answer_client.get_token_usage()["completion_tokens"] - answer_out_t +
                        self.selection_client.get_token_usage()["completion_tokens"] - sel_out_t
                    )
                    per_turn_input_tokens.append(final_answer_input_tokens)
                    per_turn_output_tokens.append(final_answer_output_tokens)
                    self.character_profile_database.close()
                    return sorted(list(final_chosen_nodes)), prediction, merged_passages, all_character_candidate, actual_turns, total_iter_input_tokens, total_iter_output_tokens, final_answer_input_tokens, final_answer_output_tokens, per_turn_input_tokens, per_turn_output_tokens

                elif decision["action"] == "EXPAND":
                    print("[Agent] Expansion triggered.")
                    expanded = set[Any]()
                    for nid, *_ in chosen_nodes:
                        if nid not in self.memory_graph.nodes:
                            print(f"[Warning] Node {nid} not in memory_graph, skip expansion for this node.")
                            continue
                        neighbors = list(self.memory_graph.neighbors(nid))
                        comm = self.memory_graph.nodes[nid].get("community", [])
                        expanded.update(neighbors)
                        expanded.update(comm)
                        expanded.add(nid)

                    candidate_pool = [(nid, self._node_timestamp(nid), self._node_text(nid), self._node_asr_periods(nid), self._node_image(nid), self._node_scene(nid)) for nid in expanded]
                    candidate_node_ids = set(nid for nid, *_ in candidate_pool)
                    character_candidate = self._extract_related_character_profiles(candidate_pool)
                    actual_turns += 1
                    turn_in_tokens = (
                        self.answer_client.get_token_usage()["prompt_tokens"] - answer_in_t +
                        self.selection_client.get_token_usage()["prompt_tokens"] - sel_in_t
                    )
                    turn_out_tokens = (
                        self.answer_client.get_token_usage()["completion_tokens"] - answer_out_t +
                        self.selection_client.get_token_usage()["completion_tokens"] - sel_out_t
                    )
                    total_iter_input_tokens += turn_in_tokens
                    total_iter_output_tokens += turn_out_tokens
                    per_turn_input_tokens.append(turn_in_tokens)
                    per_turn_output_tokens.append(turn_out_tokens)
                    break

        if not combined_evidence:
            combined_evidence = {nid: ({"text": txt, **timestamp, "asr_periods": asr_periods, "scene": scene}, img) for (nid, timestamp, txt, asr_periods, img, scene) in init_retrieved}
            all_character_candidate = self._extract_related_character_profiles(init_retrieved)
            for (nid, _, _, _, img, _) in init_retrieved:
                if self._node_level(nid) == 0 and len(fact_node_images) < MAX_IMAGES_PER_PROMPT:
                    fact_node_images.add(nid)

        print("[Agent] Max exploration turns reached without ANSWER action. Forcing answer with current evidence.")
        merged_passages, merged_images = self._merge_level0_chunks(
            list(combined_evidence.items()),
            fact_node_images
        )
        if self.with_reasoning:
            prompt = final_response_ge_multimodal_template_v2.format(
                question=question,
                context_summary=context_summary,
                character_profiles=json.dumps(character_candidate),
                passages=json.dumps(merged_passages)
            )
        else:
            prompt = final_response_ge_multimodal_template.format(
                question=question,
                context_summary=context_summary,
                character_profiles=json.dumps(character_candidate),
                passages=json.dumps(merged_passages)
            )

        image_context = {'type': 'images/jpeg', 'content': merged_images}
        text_context = {'type': 'text', 'content': prompt}
        prompt = generate_messages([text_context, image_context], system_prompt="You are an expert multimodal comprehension assistant skilled at answering questions based on video content.")
        in_before_final = self.answer_client.get_token_usage()["prompt_tokens"]
        out_before_final = self.answer_client.get_token_usage()["completion_tokens"]
        prediction = self.answer_client.obtain_response(messages=prompt, max_tokens=self.max_tokens, temperature=self.temperature).strip()
        final_answer_input_tokens = self.answer_client.get_token_usage()["prompt_tokens"] - in_before_final
        final_answer_output_tokens = self.answer_client.get_token_usage()["completion_tokens"] - out_before_final
        per_turn_input_tokens.append(final_answer_input_tokens)
        per_turn_output_tokens.append(final_answer_output_tokens)

        final_chosen_nodes = combined_evidence.keys()

        self.character_profile_database.close()
        return sorted(list(final_chosen_nodes)), prediction, merged_passages, all_character_candidate, actual_turns, total_iter_input_tokens, total_iter_output_tokens, final_answer_input_tokens, final_answer_output_tokens, per_turn_input_tokens, per_turn_output_tokens

def parse_args():
    p = argparse.ArgumentParser(description="QA with Pyravid")
    p.add_argument("--dataset", type=str, default="m3bench")
    p.add_argument("--question_dir", type=str, default=None, help="./data/<dataset>/questions/")
    p.add_argument("--super_graph_dir", type=str, default="./super_graphs/super_graphs-online-llm-long-two-level/qwen_top_20/m3bench")
    p.add_argument("--super_embedding_dir", type=str, default="./super_embeddings/super_embeddings-online-llm-long-two-level/qwen_top_20/m3bench")
    p.add_argument("--vector_store_dir", type=str, default="./snapshot")
    p.add_argument("--two_level_mode", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--api_key_path", type=str, default="./config/openai_key.txt")
    p.add_argument("--answer_api_key_path", type=str, default="./config/gemini_key.txt")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--answer_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--selection_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--embedding_model", type=str, default="text-embedding-3-large", help="Embedding model to use")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_turns", type=int, default=5)
    p.add_argument("--tolerance", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=10000)
    p.add_argument("--multimodal", action="store_true")
    p.add_argument("--rerank_with_selection", action="store_true")
    p.add_argument("--with_reasoning", action="store_true")
    p.add_argument("--with_prune", action="store_true")

    p.add_argument("--data", type=str, default="web", choices=["web", "robot"], help="Specify the dataset type to determine memory paths")

    p.add_argument("--save_dir", type=str, default="./output/output-online-llm-long-two-level/qwen_top_20_timestamp_asr")
    p.add_argument("--save_name", type=str, default=None, help="Override default output filename")
    p.add_argument("--save_evidence", action="store_true", help="Also store retrieved passage ids/texts", default=True)
    p.add_argument("--with_top_summary", action="store_true", help="Whether to include the top-level summary in the prompts")
    p.add_argument("--answer_ip_address", type=str)
    p.add_argument("--selection_ip_address", type=str)
    p.add_argument("--latency_file_name", type=str)
    return p.parse_args()

def main():

    args = parse_args()
    logging.info(f"Arguments: {args}")
    dataset = args.dataset
    data = args.data
    with_top_summary = args.with_top_summary
    question_path = args.question_dir or f'./data/{dataset}/questions/'

    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_name:
        out_path = os.path.join(args.save_dir, args.save_name)
    else:
        out_path = os.path.join(args.save_dir, f"Pyravid_QA_Output (Open Setting).json")

    prediction_json = []
    total_num = 0
    correct_num = 0

    files = [f for f in os.listdir(question_path) if f.endswith(".json")]
    files.sort()
    files = files[:10]


    per_question_latencies = []
    per_question_latencies_rows = []
    per_question_turns = []
    per_question_iter_input_tokens = []
    per_question_iter_output_tokens = []
    per_question_final_answer_input_tokens = []
    per_question_final_answer_output_tokens = []

    for file_name in files:
        video_id = os.path.splitext(file_name)[0]
        print(f"\n=== Answering Questions in {video_id} ===")

        with open(os.path.join(question_path, file_name), "r") as f:
            question_list = json.load(f)

        total_num += len(question_list)
        local_correct = 0

        for qd in tqdm(question_list, leave=False):
            qid = qd.get("QID")
            question = qd["Question"]
            answer = qd.get("Answer")
            gold = qd.get("Gold")
            timestamp = qd.get("TimeStamp", "")
            before_clip = qd.get("BeforeClip")
            if data == "web":
                gexf = f"./{args.super_graph_dir}/{dataset}/{video_id}/{video_id}_graph_level_all.gexf"
                npy = f"./{args.super_embedding_dir}/{dataset}/{video_id}/{video_id}_embedding_level_all.npy"
                facts_path = f"./{os.environ.get('PYRAVID_FACTS_DIR')}/{video_id}.json"
                with open(facts_path, "r", encoding="utf-8") as f:
                   facts_data = json.load(f)
                max_clip = max(int(k) for k in facts_data.keys())
                db = f"./{args.vector_store_dir}/snapshot/{video_id}/clip_{max_clip}_snapshot"
            elif data == "robot":
                gexf = f"./{args.super_graph_dir}/{dataset}/{video_id}/{video_id}_graph_level_all_{before_clip}.gexf"
                npy = f"./{args.super_embedding_dir}/{dataset}/{video_id}/{video_id}_embedding_level_all_{before_clip}.npy"
                db = f"./{args.vector_store_dir}/snapshot/{video_id}/clip_{before_clip}_snapshot"
            if not (os.path.exists(gexf) and os.path.exists(npy) and os.path.exists(db)):
                print(f"[Skip] No reading memory found for {video_id}.")
                print(f"  gexf exists: {os.path.exists(gexf)} -> {gexf}")
                print(f"  npy exists: {os.path.exists(npy)} -> {npy}")
                print(f"  db exists: {os.path.exists(db)} -> {db}")
                continue

            recall = OpenQuestionExplorer(
                dataset=dataset,
                video_id=video_id,
                before_clip_id=before_clip,
                vector_store_dir=args.vector_store_dir,
                api_key_path=args.api_key_path,
                answer_api_key_path=args.answer_api_key_path,
                model=args.model,
                answer_model=args.answer_model,
                multimodal=args.multimodal,
                embedding_model=args.embedding_model,
                top_k=args.top_k,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                super_graph_dir=args.super_graph_dir,
                super_embedding_dir=args.super_embedding_dir,
                two_level_mode=args.two_level_mode,
                selection_model=args.selection_model,
                with_top_summary=with_top_summary,
                data=args.data,
                with_reasoning=args.with_reasoning,
                rerank_with_selection=args.rerank_with_selection,
                with_prune=args.with_prune,
                answer_ip_address=args.answer_ip_address,
                selection_ip_address=args.selection_ip_address
            )

            start_time = time.perf_counter()
            (final_chosen_nodes, prediction, merged_passages, final_character_candidate,
             actual_turns,
             total_iter_input_tokens, total_iter_output_tokens,
             final_answer_input_tokens, final_answer_output_tokens,
             per_turn_input_tokens, per_turn_output_tokens) = recall.run(question=question, max_exploration_turns=args.max_turns, tolerance=args.tolerance)
            end_time = time.perf_counter()
            latency = end_time - start_time
            per_question_latencies.append(latency)
            per_question_turns.append(actual_turns)
            per_question_iter_input_tokens.append(total_iter_input_tokens)
            per_question_iter_output_tokens.append(total_iter_output_tokens)
            per_question_final_answer_input_tokens.append(final_answer_input_tokens)
            per_question_final_answer_output_tokens.append(final_answer_output_tokens)
            per_question_latencies_rows.append({
                "question": qid,
                "latency_seconds": round(latency, 6),
                "actual_turns": actual_turns,
                "total_iter_input_tokens": total_iter_input_tokens,
                "total_iter_output_tokens": total_iter_output_tokens,
                "final_answer_input_tokens": final_answer_input_tokens,
                "final_answer_output_tokens": final_answer_output_tokens,
                "per_turn_input_tokens": json.dumps(per_turn_input_tokens),
                "per_turn_output_tokens": json.dumps(per_turn_output_tokens),
            })
            print(f"[Q {qid}] latency={latency:.2f}s | turns={actual_turns} | iter_in={total_iter_input_tokens} | iter_out={total_iter_output_tokens} | final_in={final_answer_input_tokens} | final_out={final_answer_output_tokens} | per_turn_in={per_turn_input_tokens} | per_turn_out={per_turn_output_tokens}")

            print(f"Question: {question}")
            print(f"Prediction: {prediction}")
            score_llm = llm_judge_m3(
                client=recall.client,
                question=question,
                prediction=prediction,
                answer=answer,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            local_correct += score_llm
            rec = {
                "QID": qid,
                "Question": question,
                "TimeStamp": timestamp,
                "Answer": answer,
                "Prediction": prediction,
                "LLM_Judge": score_llm,
            }
            #print("Answer: ", answer)
            #print("LLM Judge: ", score_llm)
            print(rec)
            rec["ActualTurns"] = actual_turns
            rec["TotalIterInputTokens"] = total_iter_input_tokens
            rec["TotalIterOutputTokens"] = total_iter_output_tokens
            rec["FinalAnswerInputTokens"] = final_answer_input_tokens
            rec["FinalAnswerOutputTokens"] = final_answer_output_tokens
            rec["PerTurnInputTokens"] = per_turn_input_tokens
            rec["PerTurnOutputTokens"] = per_turn_output_tokens
            if args.save_evidence:
                rec["FinalCharacterCandidate"] = final_character_candidate
                rec["FinalChosenNodes"] =[{"node_id": chosen_node, "text": recall._node_text(chosen_node), "level": recall._node_level(chosen_node)} for chosen_node in final_chosen_nodes]
                rec["UsedEvidences"] = [{"node_id": node_id, "content": content} for node_id, content in zip(final_chosen_nodes, merged_passages)]
            prediction_json.append(rec)

        acc = local_correct / max(1, len(question_list))
        print(f"[{video_id}] QA accuracy: {acc:.4f}")
        correct_num += local_correct
        print("=" * 25)

    if total_num == 0:
        print("[Warn] No questions processed.")
    else:
        print(f"\n== Final QA Accuracy: {correct_num / total_num:.4f} ==")

    mean_turns = 0.0
    mean_input_tokens_per_iter = 0.0
    mean_output_tokens_per_iter = 0.0
    mean_final_ans_input_tokens = 0.0
    mean_final_ans_output_tokens = 0.0
    if per_question_turns:
        mean_turns = mean(per_question_turns)
        any_turn = any(n > 0 for n in per_question_turns)
        mean_input_tokens_per_iter = (
            mean(t / n for t, n in zip(per_question_iter_input_tokens, per_question_turns) if n > 0)
            if any_turn else 0.0
        )
        mean_output_tokens_per_iter = (
            mean(t / n for t, n in zip(per_question_iter_output_tokens, per_question_turns) if n > 0)
            if any_turn else 0.0
        )
        mean_final_ans_input_tokens = mean(per_question_final_answer_input_tokens)
        mean_final_ans_output_tokens = mean(per_question_final_answer_output_tokens)
        print(f"== Mean EXPAND turns per run:                {mean_turns:.2f} ==")
        print(f"== Mean LLM input tokens per iteration:      {mean_input_tokens_per_iter:.1f} ==")
        print(f"== Mean LLM output tokens per iteration:     {mean_output_tokens_per_iter:.1f} ==")
        print(f"== Mean LLM input tokens for final answer:   {mean_final_ans_input_tokens:.1f} ==")
        print(f"== Mean LLM output tokens for final answer:  {mean_final_ans_output_tokens:.1f} ==")

    os.makedirs(args.save_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(prediction_json, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_path}")

    avg_latency = 0.0
    p50_latency = 0.0
    p95_latency = 0.0
    if per_question_latencies:
        sorted_latencies = sorted(per_question_latencies)
        avg_latency = mean(per_question_latencies)
        p50_latency = _percentile(sorted_latencies, 0.50)
        p95_latency = _percentile(sorted_latencies, 0.95)

    csv_path = os.path.join(args.save_dir, args.latency_file_name)
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    fieldnames = [
        "question", "latency_seconds", "actual_turns",
        "total_iter_input_tokens", "total_iter_output_tokens",
        "final_answer_input_tokens", "final_answer_output_tokens",
        "per_turn_input_tokens", "per_turn_output_tokens",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_question_latencies_rows)
        writer.writerow({"question": "__summary_avg__",                          "latency_seconds": round(avg_latency, 6)})
        writer.writerow({"question": "__summary_p50__",                          "latency_seconds": round(p50_latency, 6)})
        writer.writerow({"question": "__summary_p95__",                          "latency_seconds": round(p95_latency, 6)})
        writer.writerow({"question": "__summary_mean_turns__",                   "latency_seconds": round(mean_turns, 2)})
        writer.writerow({"question": "__summary_mean_input_tokens_per_iter__",   "latency_seconds": round(mean_input_tokens_per_iter, 1)})
        writer.writerow({"question": "__summary_mean_output_tokens_per_iter__",  "latency_seconds": round(mean_output_tokens_per_iter, 1)})
        writer.writerow({"question": "__summary_mean_final_ans_input_tokens__",  "latency_seconds": round(mean_final_ans_input_tokens, 1)})
        writer.writerow({"question": "__summary_mean_final_ans_output_tokens__", "latency_seconds": round(mean_final_ans_output_tokens, 1)})

if __name__ == "__main__":
    main()
