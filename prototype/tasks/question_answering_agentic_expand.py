import csv
import os
import ast
import json
import argparse
import logging
import time
from statistics import mean

import openai
from typing_extensions import override

import networkx as nx
import warnings
from typing import List, Tuple, Optional
from tqdm import tqdm

from prototype.tools.prompts import (
    final_response_mc_multimodal_template_v6,
    final_response_mc_multimodal_template,
    final_response_mc_multimodal_template_v2,
    final_response_mc_multimodal_without_summary_template,
    passage_selection_v3_without_character_profiles_template,
    passage_selection_v3_without_summary_template,
    agentic_expand_template_v3,
    agentic_expand_template_v2,
    agentic_expand_without_summary_template_v2,
    final_response_mc_multimodal_template_v6_without_summary
)
from prototype.tools.utils import generate_messages, extract_prediction, extract_agentic_action
from prototype.tasks.explorer import Explorer

warnings.filterwarnings("ignore", category=UserWarning)

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


class MultipleChoiceExplorer(Explorer):
    """
    Prune-and-Grow over the memory hierarchy.

    Steps:
      1) Cosine top-k over all-level embeddings → initial candidates S.
      2) LLM selects useful subset X from S.
      3) Expand X with same-level neighbors + lower-level community members.
      4) Re-select on expanded pool; repeat until stable or reaching max turns.
      5) Merge contiguous level-0 chunks; answer with LLM.
    """

    def __init__(
            self,
            dataset: str,
            video_id: str,
            super_graph_dir: str,
            super_embedding_dir: str,
            api_key_path: str = "openai_key.txt",
            answer_api_key_path: str = "gemini_key.txt",
            model: str = "gpt-4o-mini",
            answer_model: str = "gemini-2.0-flash",
            multimodal: bool = True,
            embedding_model: str = "text-embedding-3-large",
            top_k: int = 10,
            temperature: float = 0.0,
            max_tokens: int = 1024,
            two_level_mode: bool = True,
            selection_model: str = "gemini-2.0-flash",
            with_prune: bool = True,
            with_expand: bool = True,
            with_top_summary: bool = True,
            answer_ip_address: str = "localhost",
            selection_ip_address: str = "localhost"
    ):
        super().__init__(dataset, video_id, super_graph_dir, super_embedding_dir, api_key_path, answer_api_key_path,
                         model, answer_model, multimodal, embedding_model, top_k, temperature, max_tokens,
                         two_level_mode, selection_model=selection_model, answer_ip_address=answer_ip_address, selection_ip_address=selection_ip_address)
        self.with_prune = with_prune
        self.with_expand = with_expand
        self.with_top_summary = with_top_summary

    @override
    def _llm_selection(self, question: str, context_summary: str, candidates: List[Tuple]) -> Optional[List[int]]:
        if not candidates:
            return []
        input_candidates = {idx+1: {"text": text, **timestamp} for idx, (_, timestamp, text, asr_periods, _, scene) in enumerate(candidates)}
        if self.with_top_summary:
            prompt = passage_selection_v3_without_character_profiles_template.format(question=question,context_summary=context_summary,passages=json.dumps(input_candidates))
        else:
            prompt = passage_selection_v3_without_summary_template.format(question=question, passages=json.dumps(input_candidates))

        resp = self.selection_client.obtain_response(prompt, max_tokens=1000, temperature=self.temperature).strip()
        try:
            idx_list = ast.literal_eval(resp)  # expect a python-like list
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
            options: Optional[List[str]] = None,
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
        # Initial top-k
        print("=== Initial Node Selection ===")
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

        all_retrieved = []
        fact_node_images = set()
        combined_evidence = {}
        prediction = None
        # try:
        if self.with_expand:
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
                                batch_chosen_local_idx = self._llm_selection(question, context_summary, batch)
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
                        chosen_local_idx = self._llm_selection(question, context_summary, candidate_pool)
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
                    chosen_nodes = [candidate_pool[i] for i in chosen_local_idx]
                    print(f"LLM selected {len(chosen_nodes)} candidates: {[nid for nid, *_ in chosen_nodes][:30]}{' ...' if len(chosen_nodes) > 30 else ''}")
                else:
                    chosen_nodes = candidate_pool.copy()
                    print(f"No pruning applied, using all {len(chosen_nodes)} candidates.")

                for nid, timestamp, txt, asr, img, scene in chosen_nodes:
                    all_retrieved.append((nid, timestamp, txt, asr, img, scene))
                    if self._node_level(nid) == 0 and len(fact_node_images) < MAX_IMAGES_PER_PROMPT:
                        fact_node_images.add(nid)

                combined_evidence = {
                nid: ({"text": txt, **timestamp, "asr_periods": asr, "scene": scene}, img)
                for (nid, timestamp, txt, asr, img, scene) in all_retrieved
                }

                merged_passages, merged_images = self._merge_level0_chunks(
                    list(combined_evidence.items()),
                    fact_node_images
                )

                if self.with_top_summary:
                    prompt = agentic_expand_template_v2.format(
                        question=question,
                        context_summary=context_summary,
                        options=options,
                        passages=json.dumps(merged_passages)
                    )
                else:
                    prompt = agentic_expand_without_summary_template_v2.format(
                        question=question,
                        options=options,
                        passages=json.dumps(merged_passages)
                    )

                text_context = {'type': 'text', 'content': prompt}
                image_context = {'type': 'images/jpeg', 'content': merged_images}
                prompt = generate_messages([text_context, image_context], system_prompt="You are an expert multimodal comprehension assistant skilled at answering questions based on video content.")

                final_chosen_nodes = combined_evidence.keys()

                decision_tol = tolerance
                decision = None

                while decision_tol > 0:
                    response = self.answer_client.obtain_response(
                        messages=prompt,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    ).strip()
                    print(response)
                    decision = extract_agentic_action(response)

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
                        return sorted(list(final_chosen_nodes)), prediction, merged_passages, actual_turns, total_iter_input_tokens, total_iter_output_tokens, final_answer_input_tokens, final_answer_output_tokens, per_turn_input_tokens, per_turn_output_tokens

                    elif decision["action"] == "EXPAND":
                        print("[Agent] Expansion triggered.")
                        expanded = set()
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

                if decision["action"] == "INVALID":
                    break

        # except openai.BadRequestError as e:
        #     print(f"[Error] OpenAI API error during agentic decision: {e}, using inital retrieved candidates for final answer.")
        #     combined_evidence = {
        #     nid: ({"text": txt, **timestamp, "asr_periods": asr, "scene": scene}, img)
        #     for (nid, timestamp, txt, asr, img, scene) in init_retrieved
        #     }
        #     fact_node_images = set()
        #     for (nid, _, _, _, img, _) in init_retrieved:
        #         if self._node_level(nid) == 0 and len(fact_node_images) < MAX_IMAGES_PER_PROMPT:
        #             fact_node_images.add(nid)

        if not combined_evidence:
            combined_evidence = {
            nid: ({"text": txt, **timestamp, "asr_periods": asr, "scene": scene}, img)
            for (nid, timestamp, txt, asr, img, scene) in init_retrieved
            }
            for (nid, _, _, _, img, _) in init_retrieved:
                if self._node_level(nid) == 0 and len(fact_node_images) < MAX_IMAGES_PER_PROMPT:
                    fact_node_images.add(nid)

        if self.with_expand:
            print("[Agent] Max exploration turns reached without ANSWER action. Forcing answer with current evidence.")

        merged_passages, merged_images = self._merge_level0_chunks(
            list(combined_evidence.items()),
            fact_node_images
        )

        final_chosen_nodes = combined_evidence.keys()

        try:
            if self.with_top_summary:
                prompt = final_response_mc_multimodal_template_v6.format(
                    question=question,
                    context_summary=context_summary,
                    options=options,
                    passages=json.dumps(merged_passages)
                )
            else:
                prompt = final_response_mc_multimodal_template_v6_without_summary.format(
                    question=question,
                    options=options,
                    passages=json.dumps(merged_passages)
                )

            text_context = {'type': 'text', 'content': prompt}
            image_context = {'type': 'images/jpeg', 'content': merged_images}
            prompt = generate_messages([text_context, image_context], system_prompt="You are an expert multimodal comprehension assistant skilled at answering questions based on video content.")
            in_before_final = self.answer_client.get_token_usage()["prompt_tokens"]
            out_before_final = self.answer_client.get_token_usage()["completion_tokens"]
            response = self.answer_client.obtain_response(
                messages=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            ).strip()
            final_answer_input_tokens = self.answer_client.get_token_usage()["prompt_tokens"] - in_before_final
            final_answer_output_tokens = self.answer_client.get_token_usage()["completion_tokens"] - out_before_final
            per_turn_input_tokens.append(final_answer_input_tokens)
            per_turn_output_tokens.append(final_answer_output_tokens)
            prediction = extract_prediction(response)
        except Exception as e:
            print(f"Error extracting prediction: {e}, using fallback.")
            if self.with_top_summary:
                prompt = final_response_mc_multimodal_template.format(
                question=question,
                context_summary=context_summary,
                options=options,
                passages=json.dumps(merged_passages)
                )
            else:
                prompt = final_response_mc_multimodal_without_summary_template.format(
                question=question,
                options=options,
                passages=json.dumps(merged_passages)
                )
            text_context = {'type': 'text', 'content': prompt}

            image_context = {'type': 'images/jpeg', 'content': merged_images}
            prompt = generate_messages([text_context, image_context], system_prompt="You are an expert multimodal comprehension assistant skilled at answering questions based on video content.")
            in_before_final = self.answer_client.get_token_usage()["prompt_tokens"]
            out_before_final = self.answer_client.get_token_usage()["completion_tokens"]
            prediction = self.answer_client.obtain_response(messages=prompt, max_tokens=self.max_tokens, temperature=self.temperature).strip()
            final_answer_input_tokens = self.answer_client.get_token_usage()["prompt_tokens"] - in_before_final
            final_answer_output_tokens = self.answer_client.get_token_usage()["completion_tokens"] - out_before_final
            per_turn_input_tokens.append(final_answer_input_tokens)
            per_turn_output_tokens.append(final_answer_output_tokens)

        return sorted(list(final_chosen_nodes)), prediction, merged_passages, actual_turns, total_iter_input_tokens, total_iter_output_tokens, final_answer_input_tokens, final_answer_output_tokens, per_turn_input_tokens, per_turn_output_tokens

    @override
    def _node_text(self, node_id: int) -> str:
        n = int(node_id)
        # Handle both clip level and fact level nodes
        if n not in self.memory_graph.nodes:
            print(f"[Warning] Node {n} not in memory_graph, skip.")
            return ""
        txt = self.memory_graph.nodes[n]["raw_fact_text"] if "raw_fact_text" in self.memory_graph.nodes[n] else \
        self.memory_graph.nodes[n]["text"]
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

# FIXME: We could move this function into a separate file
def parse_args():
    p = argparse.ArgumentParser(description="QA with Pyravid")
    p.add_argument("--dataset", type=str, default="videomme-test")
    p.add_argument("--question_dir", type=str, default=None, help="./data/<dataset>/questions/")
    p.add_argument("--super_graph_dir", type=str,
                   default="./super_graphs/super_graphs-online-llm-long-two-level/qwen_top_20")
    p.add_argument("--super_embedding_dir", type=str,
                   default="./super_embeddings/super_embeddings-online-llm-long-two-level/qwen_top_20")
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
    p.add_argument("--multimodal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--with_prune", action="store_true")
    p.add_argument("--data", type=str, default="videomme")
    p.add_argument("--plain_memory", action="store_true")
    p.add_argument("--with_expand", action="store_true")
    p.add_argument("--with_top_summary", action="store_true")
    p.add_argument("--latency_file_name", type=str, default="answer_latencies.csv")
    p.add_argument("--save_dir", type=str, default="./output/output-online-llm-long-two-level/qwen_top_20/lvbench/")
    p.add_argument("--save_name", type=str, default=None, help="Override default output filename")
    p.add_argument("--save_evidence", action="store_true", help="Also store retrieved passage ids/texts", default=True)
    p.add_argument("--answer_ip_address", type=str)
    p.add_argument("--selection_ip_address", type=str)
    return p.parse_args()


def main():
    args = parse_args()
    logging.info(f"Arguments: {args}")
    dataset = args.dataset
    question_path = args.question_dir or f'./data/{dataset}/questions/'

    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_name:
        out_path = os.path.join(args.save_dir, args.save_name)
    else:
        out_path = os.path.join(args.save_dir, f"Pyravid_QA_Output (MC setting).json")

    prediction_json = []
    total_num = 0
    correct_num = 0

    files = [f for f in os.listdir(question_path) if f.endswith(".json")]
    files.sort()

    per_question_latencies = []
    per_question_latencies_rows = []
    per_question_turns = []
    per_question_iter_input_tokens = []
    per_question_iter_output_tokens = []
    per_question_final_answer_input_tokens = []
    per_question_final_answer_output_tokens = []

    for file_name in files:
        print(files)
        print(question_path)
        video_id = os.path.splitext(file_name)[0]
        print(f"\n=== Answering Questions in {video_id} ===")

        # Ensure reading memory exists


        if args.plain_memory:
            print(f"[Info] Using plain memory for {video_id}.")
            if args.data == "videomme":
                path = f"./data/videomme_new/facts/{video_id}.json"
            elif args.data == "lvbench":
                path = f"./data/lvbench_new/facts/{video_id}.json"
            with open(path, "r", encoding="utf-8") as f:
                facts_data = json.load(f)
            max_clip = max(int(k) for k in facts_data.keys())
            gexf = f"./{args.super_graph_dir}/{dataset}/{video_id}/{video_id}_graph_level_0_{max_clip}.gexf"
            npy = f"./{args.super_embedding_dir}/{dataset}/{video_id}/{video_id}_embedding_level_0_{max_clip}.npy"
        else:
            gexf = f"./{args.super_graph_dir}/{dataset}/{video_id}/{video_id}_graph_level_all.gexf"
            npy = f"./{args.super_embedding_dir}/{dataset}/{video_id}/{video_id}_embedding_level_all.npy"
        print(f"[Info] Checking for reading memory files: {gexf}, {npy}")
        if not (os.path.exists(gexf) and os.path.exists(npy)):
                print(f"[Skip] No reading memory found for {video_id}.")
                continue

        recall = MultipleChoiceExplorer(
            dataset=dataset,
            video_id=video_id,
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
            with_prune=args.with_prune,
            with_expand=args.with_expand,
            with_top_summary=args.with_top_summary,
            answer_ip_address=args.answer_ip_address,
            selection_ip_address=args.selection_ip_address
        )

        with open(os.path.join(question_path, file_name), "r") as f:
            question_list = json.load(f)

        total_num += len(question_list)
        local_correct = 0

        for qd in tqdm(question_list, leave=False):
            qid = qd.get("QID")
            question = qd["Question"]
            answer = qd.get("Answer")
            options = qd.get("Options")
            gold = qd.get("Gold")
            aspect = qd.get("Aspect")
            complexity = qd.get("Complexity")

            start_time = time.perf_counter()
            (final_chosen_nodes, prediction, used_passages, actual_turns,
             total_iter_input_tokens, total_iter_output_tokens,
             final_answer_input_tokens, final_answer_output_tokens,
             per_turn_input_tokens, per_turn_output_tokens) = recall.run(
                question=question,
                options=options,
                max_exploration_turns=args.max_turns,
                tolerance=args.tolerance,
            )
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

            is_correct = (str(prediction).strip().lower() == str(gold).strip().lower())
            local_correct += 1 if is_correct else 0
            rec = {
                "QID": qid,
                "Aspect": aspect,
                "Complexity": complexity,
                "Question": question,
                "Options": options,
                "Gold": gold,
                "Prediction": prediction,
                "Correct": is_correct,
            }
            print(f"Gold: {gold}")
            print("Result: ", "Correct" if is_correct else "Incorrect")
            rec["ActualTurns"] = actual_turns
            rec["TotalIterInputTokens"] = total_iter_input_tokens
            rec["TotalIterOutputTokens"] = total_iter_output_tokens
            rec["FinalAnswerInputTokens"] = final_answer_input_tokens
            rec["FinalAnswerOutputTokens"] = final_answer_output_tokens
            rec["PerTurnInputTokens"] = per_turn_input_tokens
            rec["PerTurnOutputTokens"] = per_turn_output_tokens
            if args.save_evidence:
                rec["FinalChosenNodes"] = [{"node_id": chosen_node, "text": recall._node_text(chosen_node),
                                            "level": recall._node_level(chosen_node)} for chosen_node in
                                           final_chosen_nodes]
                rec["UsedEvidences"] = [{"node_id": node_id, "content": content} for node_id, content in
                                        zip(final_chosen_nodes, used_passages)]
            prediction_json.append(rec)

        acc = local_correct / max(1, len(question_list))
        correct_num += local_correct
        print(f"[{video_id}] QA accuracy (MC): {acc:.4f}")
        print("=" * 25)

    # Summary
    if total_num == 0:
        print("[Warn] No questions processed.")
    else:
        print(f"\n== Final MC accuracy: {correct_num / total_num:.4f} ==")

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

    # Save
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

    csv_file_name = args.latency_file_name

    csv_path = os.path.join(args.save_dir, csv_file_name)
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
