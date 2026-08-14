from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json_repair
import ast
import numpy as np
import warnings
import json
import argparse
import re
import csv
from prototype.tools.api_client import APIClient
from prototype.tools.search_utils import MultiSearcher
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from prototype.tools.prompts import text_summarization_template, highlevel_memory_summarization_template, link_generation_template_v2
from itertools import combinations
from multiprocessing import Pool, cpu_count
from prototype.tools.utils import generate_messages, smart_json_loads, generate_qwen_messages
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
MAX_RETRIES = 10

def parse_args():
    p = argparse.ArgumentParser(
        description="One-shot Constructivist Agentic Memory with hierarchy (parallel single-doc).")
    p.add_argument("--dataset", type=str,
                   default="lvbench_new", help="Dataset name.")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="Edge activation threshold.")
    p.add_argument("--weight", type=float, default=0.6,
                   help="Weight for text similarity vs proximity (0~1).")
    p.add_argument("--sigma", type=float, default=1.0,
                   help="Sigma for Gaussian proximity similarity.")
    p.add_argument("--k", type=int, default=20,
                   help="Top-k neighbors per node.")
    p.add_argument("--max_cluster_size", type=int, default=12,
                   help="Maximum nodes allowed in one community.")
    p.add_argument("--api_key_path", type=str,
                   default="./config/openai_key.txt", help="Path to OpenAI API key.")
    p.add_argument("--link_api_key_path", type=str,
                   default="./config/gemini_key.txt", help="Path to Gemini API key.")
    p.add_argument("--model", type=str, default="gemini-2.0-flash",
                   help="LLM model for summarization.")
    p.add_argument("--online_mode", action=argparse.BooleanOptionalAction,
                   help="generate memory in online setting", default=True)
    p.add_argument("--embedding_model", type=str,
                   default="text-embedding-3-large", help="Embedding model to use")
    p.add_argument("--link_model", type=str, default="Qwen/Qwen3-32B",
                   help="LLM model for link generation.")
    p.add_argument("--llm_link", action=argparse.BooleanOptionalAction,
                   help="Use LLM to generate links.", default=True)
    p.add_argument("--two_level_mode", action=argparse.BooleanOptionalAction,
                   help="Use two-level mode.", default=True)
    p.add_argument("--max_hierarchy_level", type=int,
                   default=10, help="Maximum hierarchy levels.")
    p.add_argument("--summary_field", type=str, default="raw_fact_text",
                   help="Node attribute to use as text for hierarchical summarization.")
    p.add_argument("--facts_dir", type=str,
                   default="./data/lvbench_new/facts", help="Directory to save facts.")
    p.add_argument("--super_graph_dir", type=str,
                   default="./memory/graphs", help="Directory to save super graph.")
    p.add_argument("--super_embedding_dir", type=str,
                   default="./memory/embeddings", help="Directory to save super embedding.")
    p.add_argument("--vector_store_dir", type=str,
                   default="./vectorstore", help="Directory to save vector store.")
    p.add_argument("--num_processes", type=int, default=1,
                   help="Processes for single-doc parallel build (default=min(10,CPU)).")
    p.add_argument("--dense_top_k", type=int, default=50,
                   help="Top-k results for dense cosine search channel.")
    p.add_argument("--sparse_top_k", type=int, default=50,
                   help="Top-k results for BM25 sparse search channel.")
    p.add_argument("--rrf_k", type=int, default=60,
                   help="RRF fusion constant (higher = smoother blending).")
    p.add_argument("--enable_entity_relation_overlap", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable LLM-based entity and relation overlap as additional search channels.")
    p.add_argument("--entity_top_k", type=int, default=50,
                   help="Top-k results for entity overlap search channel. Defaults to sparse_top_k when unset.")
    p.add_argument("--relation_top_k", type=int, default=50,
                   help="Top-k results for relation overlap search channel. Defaults to sparse_top_k when unset.")
    p.add_argument("--entity_relation_alpha", type=float, default=0.5,
                   help="Reserved tuning weight for entity vs relation overlap.")
    p.add_argument("--qwen", action="store_true", help="Use local Qwen model for memory construction.")
    p.add_argument(
        "--qwen_server",
        action=argparse.BooleanOptionalAction,
        help="Use Qwen server for memory construction.",
        default=True,
    )
    return p.parse_args()


class CAM:
    def __init__(self,
                 dataset: str,
                 super_graph_dir: str,
                 super_embedding_dir: str,
                 vector_store_dir: str,
                 threshold: float = 0.6,
                 weight: float = 0.6,
                 sigma: float = 1.0,
                 top_k: int = 10,
                 dense_top_k: int = 20,
                 sparse_top_k: int = 20,
                 rrf_k: int = 60,
                 enable_entity_relation_overlap: bool = False,
                 entity_top_k: int | None = None,
                 relation_top_k: int | None = None,
                 entity_relation_alpha: float = 0.5,
                 api_key_path: str = "openai_key.txt",
                 link_api_key_path: str = "gemini_key.txt",
                 model: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-3-large",
                 link_model: str = "gpt-4o-mini",
                 llm_link: bool = True,
                 two_level_mode: bool = True,
                 max_cluster_size: int = 12,
                 summary_field: str = "text",
                 use_qwen: bool = True,
                 qwen_model: APIClient = None,
                 qwen_server: bool = True):

        self.dataset = dataset
        self.activation_threshold = threshold
        self.w_text = weight
        self.sigma = sigma
        self.top_k = top_k
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.enable_entity_relation_overlap = enable_entity_relation_overlap
        self.entity_top_k = entity_top_k
        self.relation_top_k = relation_top_k
        self.entity_relation_alpha = float(np.clip(entity_relation_alpha, 0.0, 1.0))
        self.max_cluster_size = max_cluster_size
        self.summary_field = summary_field

        self.graph_out_dir = super_graph_dir
        self.emb_out_dir = super_embedding_dir
        self.vector_store_dir = vector_store_dir

        self.use_qwen = use_qwen
        if not self.use_qwen:
            self.client = APIClient("gemini", link_api_key_path, model, None, debug=True)
            self.embedding_client = APIClient("openai", api_key_path, model, embedding_model)
            self.link_client = APIClient("gemini", link_api_key_path, link_model, None, debug=True)
        else:
            self.client = APIClient("gemini", link_api_key_path, model, embedding_model)
            self.embedding_client = APIClient("openai", api_key_path, model, embedding_model)
            if qwen_model is None and not qwen_server:
                raise ValueError("Qwen model is not provided")
            if qwen_server:
                self.link_client = APIClient("qwen-server-link-model", None, link_model, None, debug=True, think_mode=False)
            else:
                self.link_client = qwen_model

        self.llm_link = llm_link
        self.two_level_mode = two_level_mode

        self.memory = nx.Graph()
        self.clip_summary_memory = nx.Graph()
        self.highlevel_memory = None
        self.embeddings = None                  # [#facts, D]
        self.clip_summary_embeddings = None     # [#clip, D]
        # [#facts, #facts], only used in multi-doc
        self.doc_mask = None
        self.node_ids = []
        self.doc_ids = []

        # Chat-token usage stats per source:
        #   qwen   -> link_client (qwen-server / qwen) calls in _add_edges_from_llm_links
        #   gemini -> client (gemini) calls in _text_summarization + _build_or_update_highlevel_memory_state
        self.token_stats = {
            source: {"prompt_tokens": 0, "completion_tokens": 0, "num_calls": 0}
            for source in ("qwen", "gemini")
        }
        # Per-clip rows produced during build_online_memory.
        self.per_clip_token_rows = []

    def _record_tokens(self, source: str, before: dict, after: dict, num_calls: int = 1):
        for k in ("prompt_tokens", "completion_tokens"):
            self.token_stats[source][k] += int(after.get(k, 0) - before.get(k, 0))
        self.token_stats[source]["num_calls"] += int(num_calls)

    def _snapshot_stats(self) -> dict:
        return {s: dict(v) for s, v in self.token_stats.items()}

    @staticmethod
    def _diff_stats(before: dict, after: dict) -> dict:
        return {s: {k: after[s][k] - before[s][k] for k in after[s]} for s in after}

    @staticmethod
    def _strip_think_block(response: str) -> str:
        """Remove model reasoning traces wrapped in <think>...</think>."""
        if not isinstance(response, str):
            return response
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.IGNORECASE | re.DOTALL).strip()
        return cleaned if cleaned else response.strip()

    def _prepare_output_dirs(self, video_id):
        self.video_id = video_id
        self.graph_out_dir = os.path.join(self.graph_out_dir, self.dataset, video_id)
        self.emb_out_dir = os.path.join(self.emb_out_dir, self.dataset, video_id)
        os.makedirs(self.graph_out_dir, exist_ok=True)
        os.makedirs(self.emb_out_dir, exist_ok=True)

    def _save_level_graph_and_embeddings(self, graph: nx.Graph, embeddings: np.ndarray, level: int, before_clip_id: int = None):
        """Save graph & embeddings for a specific level; also remember them in level arrays."""
        # Keep in memory for composing later
        self.level_graphs.append(graph.copy())
        self.level_embeddings.append(
            embeddings.copy() if embeddings is not None else None)

        # Persist to disk
        if embeddings is not None:
            if before_clip_id is not None:
                np.save(os.path.join(
                    self.emb_out_dir, f"{self.video_id}_embedding_level_{level}_{before_clip_id}.npy"), embeddings)
            else:
                np.save(os.path.join(
                    self.emb_out_dir, f"{self.video_id}_embedding_level_{level}.npy"), embeddings)

        G = graph.copy()
        for node, data in G.nodes(data=True):
            for k, v in data.items():
                if isinstance(v, list):
                    G.nodes[node][k] = str(v)

        if before_clip_id is not None:
            nx.write_gexf(G, os.path.join(
                self.graph_out_dir, f"{self.video_id}_graph_level_{level}_{before_clip_id}.gexf"))
        else:
            nx.write_gexf(G, os.path.join(self.graph_out_dir,
                          f"{self.video_id}_graph_level_{level}.gexf"))

    def _save_graph_with_community(self, graph, community_dict, level):
        """Save graph with 'community' node attribute (for Gephi inspection)."""
        G = graph.copy()
        nx.set_node_attributes(G, community_dict, "community")
        for node, data in G.nodes(data=True):
            for k, v in data.items():
                if isinstance(v, list):
                    G.nodes[node][k] = str(v)
        out_path = os.path.join(
            self.graph_out_dir, f"{self.video_id}_graph_with_community_{level}.gexf")
        nx.write_gexf(G, out_path)

    def _add_nodes_from_metadata(self, metadata: list[dict]):
        """
        Metadata list entries should contain:
        - fact_id (int | str)
        - raw_fact_text (str)
        - character_level_facts (str)
        - image_path (str): optional
        - fact_uuid (str)
        - clip_id (int | str)
        - timestamp (str)
        """
        N = len(metadata)
        start_node_id = self.memory.number_of_nodes()
        new_node_ids = list(range(start_node_id, start_node_id + N))
        self.node_ids.extend(new_node_ids)
        clip_level_metainfos = []
        for _, item in enumerate(metadata):
            fact_id = int(item.get("fact_id"))
            fact_uuid = item.get("fact_uuid")
            clip_id = int(item.get("clip_id"))
            timestamp = item.get("timestamp")
            if fact_id == None:
                raise ValueError(f"fact_id is None for fact {item}")
            if clip_id == None:
                raise ValueError(f"clip_id is None for fact {item}")
            metainfo = dict(
                fact_id=fact_id,
                raw_fact_text=item.get("raw_fact_text", ""),
                character_level_facts=item.get("character_level_facts", ""),
                scene_fact_description=item.get("scene_description", ""),
                asr_periods=item.get("asr_periods", []),
                fact_uuid=fact_uuid,
                image_path=item.get("image_path", ""),
                clip_id=clip_id,
                timestamp=timestamp
            )
            self.memory.add_node(
                fact_id,
                **metainfo
            )
            clip_level_metainfos.append(metainfo)
        if self.llm_link:
            self.vector_store.insert(self.embeddings[start_node_id:start_node_id + N, :].tolist(
            ), payloads=clip_level_metainfos, ids=new_node_ids)
        return new_node_ids, metadata[0]["timestamp"], metadata[-1]["timestamp"]

    def _infer_node_links(self, facts: list[dict]):
        for fact in facts:
            query_node_id = fact.get("fact_id")
            if query_node_id is None:
                raise ValueError(f"fact_id is None for fact {fact}")
            query_embedding = self.embeddings[query_node_id, :].tolist()
            query_text = fact.get("raw_fact_text", "")
            hits = self.vector_store._search_dense(
                query_vector=query_embedding,
                limit=self.top_k,
            )
            id2facts = []
            for hit in hits:
                node_id = hit.payload.get("fact_id")
                if node_id is None:
                    raise ValueError(f"fact_id is None for fact {hit}")
                if node_id == query_node_id:
                    continue
                id2fact = {"node_id": node_id, "text": hit.payload.get("raw_fact_text"), "timestamp": hit.payload.get("timestamp")}
                id2facts.append(id2fact)
            query_fact = {"node_id": query_node_id, "text": fact.get("raw_fact_text"), "timestamp": fact.get("timestamp")}
            self._add_edges_from_llm_links(
                query_node_id, json.dumps(query_fact), id2facts)

    def _add_edges_from_llm_links(self, query_node_id: int, query_fact_json: str, facts_list: list[dict]):
        batch_size = 5
        if len(facts_list) <= batch_size:
            fact_batches = [facts_list]
        else:
            fact_batches = [facts_list[i:i + batch_size]
                            for i in range(0, len(facts_list), batch_size)]

        allowed_categories = {"temporal", "causal", "same_event"}
        for batch_idx, fact_batch in enumerate(fact_batches, start=1):
            batch_target_node_ids = set()
            for fact in fact_batch:
                try:
                    batch_target_node_ids.add(int(fact["node_id"]))
                except Exception as e:
                    print(f"Error parsing batch target id in batch {batch_idx}: {e}")
                    continue

            batch_facts_list_json = json.dumps(fact_batch)
            prompt = link_generation_template_v2.format(
                query_fact_json=query_fact_json, facts_list_json=batch_facts_list_json)
            if self.use_qwen:
                prompt = generate_qwen_messages(
                    prompt, system_prompt="You are an expert in graph memory construction.")
            else:
                prompt = generate_messages([{"type": "text", "content": prompt}],
                                           system_prompt="You are an expert in graph memory construction.")

            json_response = None
            for attempt in range(MAX_RETRIES):
                try:
                    before = self.link_client.get_token_usage()
                    response = self.link_client.obtain_response(messages=prompt, max_tokens=10000, temperature=0.7)
                    self._record_tokens("qwen", before, self.link_client.get_token_usage(), num_calls=1)
                    response = self._strip_think_block(response)
                    repaired_response = json_repair.repair_json(response)
                    json_response = ast.literal_eval(repaired_response)
                    if json_response is not None:
                        break
                except Exception as e:
                    print(f"Error generating links for batch {batch_idx}: {e}")
                    if attempt == MAX_RETRIES - 1:
                        print(
                            f"Skip batch {batch_idx}: failed to generate links after {MAX_RETRIES} attempts")

            if json_response is None:
                continue
            if 'links' not in json_response:
                print(f"Skip batch {batch_idx}: no links found in the response")
                continue

            for link in json_response['links']:
                try:
                    target_node_id = int(link['target'])
                    category = str(link.get("category", "")).strip().lower()
                    if category not in allowed_categories:
                        raise ValueError(
                            f"Invalid link category '{category}'. Allowed: {sorted(allowed_categories)}")
                    if target_node_id in batch_target_node_ids:
                        self.memory.add_edge(
                            query_node_id, target_node_id, category=category)
                    else:
                        raise ValueError(
                            f"Invalid target node id:{target_node_id}. The target must come from the current batch.")
                except Exception as e:
                    print(f"Error adding edge: {e}")
                    continue

    def _build_pairwise_similarity(self):
        """
        Build N x N similarity with text + proximity.
        In multi-doc mode, proximity is masked by doc_mask (cross-doc = 0).
        """
        # extract embeddings for currently existing nodes in memory
        N = len(self.node_ids)
        embeddings = self.embeddings[:N, :]

        # Textual similarity
        text_sim = cosine_similarity(embeddings, embeddings)
        np.fill_diagonal(text_sim, 0.0)
        text_sim = np.maximum(0, text_sim)

        # Proximity based on chunk_id (within the same doc)
        fact_ids = np.array([self.memory.nodes[n]['fact_id']
                            for n in self.node_ids], dtype=np.int64)
        diff = np.abs(fact_ids[:, None] - fact_ids[None, :])
        proximity_sim = np.exp(- (diff ** 2) / (2 * (self.sigma ** 2)))
        np.fill_diagonal(proximity_sim, 0.0)

        # If doc_mask is present (multi-doc), zero out cross-doc proximity
        if self.doc_mask is not None:
            if self.doc_mask.shape != (N, N):
                raise ValueError(
                    f"doc_mask shape {self.doc_mask.shape} mismatches N={N}.")
            proximity_sim *= self.doc_mask

        # Weighted combination
        sim = self.w_text * text_sim + (1.0 - self.w_text) * proximity_sim
        return sim

    def _add_edges_from_similarity(self, sim: np.ndarray):
        """
        For each node, connect to top-k neighbors above threshold.
        """
        N = sim.shape[0]
        for i in range(N):
            sims = sim[i]
            above = np.where(sims >= self.activation_threshold)[0]
            if above.size == 0:
                continue
            # stable sort by descending similarity
            top_idx = above[np.argsort(-sims[above], kind='stable')
                            ][:min(self.top_k, above.size)]
            for j in top_idx:
                if i != j:
                    self.memory.add_edge(i, j, weight=float(sims[j]))
        self.memory.remove_edges_from(nx.selfloop_edges(self.memory))

    def _limit_community_size(self, communities):
        """Split large communities if exceeding max_cluster_size."""
        if not self.max_cluster_size:
            return communities
        new_comms = []
        for comm in communities:
            comm = list(comm)
            if len(comm) <= self.max_cluster_size:
                new_comms.append(set(comm))
            else:
                for i in range(0, len(comm), self.max_cluster_size):
                    new_comms.append(set(comm[i:i + self.max_cluster_size]))
        return new_comms

    def _create_persona_graph(self, graph: nx.Graph):
        components = {}
        personalities = {}
        index = 0

        # Step 1: Create egonets and partition them
        for node in graph.nodes():
            # Egonet minus ego: subgraph of neighbors
            if not list(graph.neighbors(node)):
                personas = [index]
                index += 1
                personalities[node] = personas
                continue
            ego_net_minus_ego = graph.subgraph(graph.neighbors(node))
            comps = list(nx.connected_components(ego_net_minus_ego))
            new_mapping = {}
            personas = []
            for comp in comps:
                personas.append(index)
                for other_node in comp:
                    new_mapping[other_node] = index
                index += 1
            components[node] = new_mapping  # {node: {neighbor: persona_id}}
            personalities[node] = personas  # {node: [persona_ids]}

        # Step 2: Map edges to persona graph
        persona_graph_edges = []
        for u, v in graph.edges():
            if v in components[u] and u in components[v]:
                persona_u_v = components[u][v]  # Persona of v in u's egonet
                persona_v_u = components[v][u]  # Persona of u in v's egonet
                weight = graph[u][v].get('weight', 1.0)
                persona_graph_edges.append(
                    (persona_u_v, persona_v_u, {'weight': weight}))

        # Step 3: Create the persona graph
        persona_graph = nx.Graph()
        persona_graph.add_nodes_from(range(index))
        persona_graph.add_edges_from(persona_graph_edges)

        personality_map = {p: n for n in graph.nodes()
                           for p in personalities[n]}

        return persona_graph, components, personalities, personality_map

    def _build_super_graph(self, graph, communities, community_dict):
        super_graph = nx.Graph()
        super_graph.add_nodes_from(range(len(communities)))
        # overlap edges
        for idx1, idx2 in combinations(range(len(communities)), 2):
            if communities[idx1] & communities[idx2]:
                super_graph.add_edge(idx1, idx2)
        # non-overlap edges
        for (u, v) in graph.edges():
            for comm1 in community_dict[u]:
                for comm2 in community_dict[v]:
                    if comm1 != comm2:
                        super_graph.add_edge(comm1, comm2)
        return super_graph

    def _text_summarization(self, texts: list[str]):
        """LLM-based community summarization."""
        input_texts = "\n".join(
            f"Passage {i+1}: {text}" for i, text in enumerate(texts))
        prompt = text_summarization_template.format(input_texts=input_texts)
        before = self.client.get_token_usage()
        response = self.client.obtain_response(prompt, max_tokens=1024, temperature=0.0)
        self._record_tokens("gemini", before, self.client.get_token_usage(), num_calls=1)
        super_gist = response.strip()
        super_emb = np.array(self.embedding_client.obtain_embedding(super_gist), dtype=np.float32)
        return super_gist, super_emb

    def _print_community_stats(self, level: int, book_title: str, graph: nx.Graph, communities: list[set]):
        sizes = [len(c) for c in communities]
        num = len(sizes)
        if num == 0:
            print(f"[Level {level}] No communities found.")
            return
        max_size = max(sizes)
        min_size = min(sizes)
        avg_size = sum(sizes) / num

        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

        print(
            f"[Level {level} of {book_title}] Nodes: {num_nodes} | Edges: {num_edges} | Avg Degree: {avg_degree:.2f}")
        print(f"[Level {level} of {book_title}] Communities: {num} | Max size: {max_size} | Min size: {min_size} | Avg size: {avg_size:.2f}")

    def _build_clip_summary_graph(self, clip_id: int, clip_summary_text: str, character_level_clip_summary: str, scene_clip_summary: str, fact_ids: list[int], timestamp_start: str, timestamp_end: str):
        metainfo = dict(
            clip_id=clip_id,
            text=clip_summary_text,
            character_level_clip_summary=character_level_clip_summary,
            scene_clip_summary = scene_clip_summary,
            community=fact_ids,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end
        )
        self.clip_summary_memory.add_node(clip_id, **metainfo)
        # add cross-clip edges
        # FIXME: this could be further optimized by collecting new fact node edges in fly in _add_edges_from_llm_links
        # So that we don't need to iterate over all edges every time
        new_cross_clip_edges = []
        for u, v in self.memory.edges():
            try:
                clip_id_u = self.memory.nodes[u]['clip_id']
                clip_id_v = self.memory.nodes[v]['clip_id']
                if (u in fact_ids or v in fact_ids) and (clip_id_u != clip_id_v):
                    new_cross_clip_edges.append((clip_id_u, clip_id_v))
            except Exception as e:
                print(f"[Warn] Skip edge ({u}, {v}) due to error: {e}")
                continue
        new_cross_clip_edges = list(set(new_cross_clip_edges))
        for (u, v) in new_cross_clip_edges:
            self.clip_summary_memory.add_edge(u, v)

    def _build_or_update_highlevel_memory_state(self, new_clip_summary_text: str, clip_id: int):
        # Initialize highlevel summary with first clip summary
        if self.highlevel_memory is None:
            self.highlevel_memory = nx.Graph()
            self.highlevel_memory.add_node(
                0,
                text=self.clip_summary_memory.nodes[0]['text'] if 0 in self.clip_summary_memory.nodes else self.clip_summary_memory.nodes[1]['text'],
                community=[0] if 0 in self.clip_summary_memory.nodes else [1]
            )
        else:
            # Update highlevel summary with new clip summary
            prompt = highlevel_memory_summarization_template.format(
                highlevel_summary_text=self.highlevel_memory.nodes[0]['text'], new_clip_summary_text=new_clip_summary_text)
            for attempt in range(MAX_RETRIES):
                try:
                    prompt = generate_messages(
                        [{"type": "text", "content": prompt}], system_prompt="You are an expert video summarizer. Your task is to update a running high-level summary with details from a new video clip.")
                    before = self.client.get_token_usage()
                    response = self.client.obtain_response(
                        messages=prompt, max_tokens=10000, temperature=0.0)
                    self._record_tokens("gemini", before, self.client.get_token_usage(), num_calls=1)
                    response = response.strip()
                    break
                except Exception as e:
                    print(f"Error updating highlevel summary: {e}")
                    if attempt == MAX_RETRIES - 1:
                        raise Exception(
                            f"Failed to update highlevel summary after {MAX_RETRIES} attempts")
            self.highlevel_memory.nodes[0]['community'].append(clip_id)
            self.highlevel_memory.nodes[0]['text'] = response
        highlevel_memory_embedding = np.array(self.embedding_client.obtain_embedding(
            self.highlevel_memory.nodes[0]['text']), dtype=np.float32)
        return np.stack([highlevel_memory_embedding], axis=0)

    def run_hierarchy(self, video_id, max_hierarchy_level: int = 10):

        self.level_graphs = []
        self.level_embeddings = []

        # Level 0: current memory state
        self._save_level_graph_and_embeddings(
            graph=self.memory, embeddings=self.embeddings, level=0)

        current_graph = self.memory.copy()
        prev_num_communities = current_graph.number_of_nodes()
        level = 0

        while True:
            persona_graph, _, _, personality_map = self._create_persona_graph(
                current_graph)
            communities = list(
                nx.community.label_propagation_communities(persona_graph))
            communities = self._limit_community_size(communities)

            # map persona communities back to original nodes
            overlap_communities = []
            community_dict = {}
            seen_communities = set()  # Track unique node sets
            for idx, comm in enumerate(communities):
                original_nodes = {personality_map[pid] for pid in comm}
                # Convert to frozenset for hashing
                nodes_frozenset = frozenset(original_nodes)
                if nodes_frozenset not in seen_communities:
                    seen_communities.add(nodes_frozenset)
                    overlap_communities.append(original_nodes)
                    for n in original_nodes:
                        community_dict.setdefault(n, []).append(
                            len(overlap_communities) - 1)
                else:
                    # Find the existing community index
                    existing_idx = next(i for i, existing_comm in enumerate(overlap_communities)
                                        if frozenset(existing_comm) == nodes_frozenset)
                    for n in original_nodes:
                        if existing_idx not in community_dict.get(n, []):
                            community_dict.setdefault(
                                n, []).append(existing_idx)

            self._print_community_stats(
                level, video_id, current_graph, overlap_communities)
            self._save_graph_with_community(
                current_graph, community_dict, level)

            current_num_communities = len(overlap_communities)
            if level >= max_hierarchy_level or current_num_communities >= prev_num_communities:
                break

            level += 1

            # Build super graph for next level
            super_graph = self._build_super_graph(
                current_graph, overlap_communities, community_dict)
            node_comm_dict = {idx: sorted(list(comm))
                              for idx, comm in enumerate(overlap_communities)}
            nx.set_node_attributes(super_graph, node_comm_dict, "community")

            # Summarize chosen field per community
            new_texts = []
            new_embs = []
            for comm in tqdm(overlap_communities, desc=f"Summarizing level {level-1} communities for {video_id}"):
                node_list_sorted = sorted(list(comm))
                texts = [current_graph.nodes[n].get(
                    self.summary_field) for n in node_list_sorted]
                super_gist, emb = self._text_summarization(texts)
                new_texts.append(super_gist)
                new_embs.append(emb)
            new_embs = np.stack(new_embs, axis=0)
            nx.set_node_attributes(super_graph, {n: txt for n, txt in zip(
                super_graph.nodes(), new_texts)}, self.summary_field)

            # Save this level graph+embeddings (store level-wise, do NOT overwrite base)
            self._save_level_graph_and_embeddings(
                graph=super_graph, embeddings=new_embs, level=level)

            # Prepare next iteration
            prev_num_communities = current_num_communities
            current_graph = super_graph

        print(f"[Hierarchy] Completed up to level {level}.")

        # Compose ALL levels into a single graph and embeddings
        self._compose_all_levels()

    def _compose_all_levels(self, before_clip_id: int = None):

        composed_graphs = []
        composed_embeddings = []
        offset = 0
        prev_offset = 0
        # record start offset of each level (to convert community ids)
        level_offsets = []

        for level_idx, (g_level, emb_level) in enumerate(zip(self.level_graphs, self.level_embeddings)):
            start_offset = offset
            level_offsets.append(start_offset)

            # relabel nodes
            g = g_level.copy()
            mapping = {node: node + start_offset for node in g.nodes()}
            g = nx.relabel_nodes(g, mapping)

            # annotate
            nx.set_node_attributes(g, level_idx, "level")
            nx.set_node_attributes(g, start_offset, "offset")
            nx.set_node_attributes(g, prev_offset, "prev_offset")

            # convert community -> global ids (only for level >= 1)
            if level_idx >= 1:
                prev_level_offset = level_offsets[level_idx - 1]
                for node in g.nodes():
                    orig_node = node - start_offset  # reverse mapping to original id in g_level
                    comm_raw = g_level.nodes[orig_node].get("community")
                    comm_conv = [int(c) + prev_level_offset for c in comm_raw]
                    g.nodes[node]["community"] = comm_conv

            composed_graphs.append(g)
            composed_embeddings.append(emb_level)

            prev_offset = start_offset
            offset += g.number_of_nodes()

        # Compose graphs & concat embeddings
        G_all = nx.compose_all(composed_graphs)
        E_all = np.concatenate(composed_embeddings, axis=0)

        # Save
        G_safe = G_all.copy()
        for node, data in G_safe.nodes(data=True):
            for k, v in data.items():
                if isinstance(v, list):
                    G_safe.nodes[node][k] = str(v)
        if before_clip_id is not None:
            nx.write_gexf(G_safe, os.path.join(
                self.graph_out_dir, f"{self.video_id}_graph_level_all_{before_clip_id}.gexf"))
            np.save(os.path.join(self.emb_out_dir,
                    f"{self.video_id}_embedding_level_all_{before_clip_id}.npy"), E_all)
        else:
            nx.write_gexf(G_safe, os.path.join(self.graph_out_dir,
                          f"{self.video_id}_graph_level_all.gexf"))
            np.save(os.path.join(self.emb_out_dir,
                    f"{self.video_id}_embedding_level_all.npy"), E_all)

        # Expose as current memory
        if before_clip_id is None:
            self.memory = G_all
            self.embeddings = E_all
            print(
                f"[Compose] All levels composed: nodes={self.memory.number_of_nodes()}, edges={self.memory.number_of_edges()}")

    def _prepare_vector_store(self, video_id: str):
        if not self.llm_link:
            print(
                "[Warning] LLM-link is not enabled. Using pairwise similarity to build edges.")
            return
        vector_store_path = f'{self.vector_store_dir}/vector_store/qdrant'
        os.makedirs(vector_store_path, exist_ok=True)
        self.vector_store = MultiSearcher(
            collection_name=video_id,
            path=f'{vector_store_path}/{video_id}',
            embedding_model_dims=self.embeddings.shape[1],
            dense_top_k=self.dense_top_k,
            sparse_top_k=self.sparse_top_k,
            rrf_k=self.rrf_k,
            entity_client=self.client,
            enable_entity_relation_overlap=self.enable_entity_relation_overlap,
            entity_top_k=self.entity_top_k,
            relation_top_k=self.relation_top_k,
            entity_relation_alpha=self.entity_relation_alpha,
        )

    def build_online_memory(self,
                            video_id,
                            fact_embeddings,
                            clip_summary_embeddings,
                            metadata,
                            max_hierarchy_level: int = 10):
        self._prepare_output_dirs(video_id)
        self.embeddings = np.atleast_2d(fact_embeddings).astype(np.float32)
        self.clip_summary_embeddings = np.atleast_2d(
            clip_summary_embeddings).astype(np.float32)
        if self.llm_link:
            self._prepare_vector_store(video_id)

        # Reset clients so per-video token accounting is clean.
        self.client.reset_token_usage()
        self.link_client.reset_token_usage()
        self.embedding_client.reset_token_usage()
        self.per_clip_token_rows = []

        for clip_id, facts in tqdm(metadata.items(), desc=f"Building online memory for {video_id}"):
            stats_before = self._snapshot_stats()
            new_fact_node_ids, timestamp_start, timestamp_end = self._add_nodes_from_metadata(facts['facts'])
            # FIXME: Currently this execution path is not considered
            if not self.llm_link:
                sim = self._build_pairwise_similarity()
                self._add_edges_from_similarity(sim)
            else:
                self._infer_node_links(facts['facts'])
                if self.two_level_mode:
                    # Clear intermediate memeory states for subsequent composition
                    self.level_graphs = []
                    self.level_embeddings = []
                    # Save current fact memory state
                    self._save_level_graph_and_embeddings(
                        graph=self.memory, embeddings=self.embeddings[:new_fact_node_ids[-1] + 1, :], level=0, before_clip_id=clip_id)
                    # Build clip summary graph
                    self._build_clip_summary_graph(
                        int(clip_id), facts['clip_summary'], facts['character_level_clip_summary'], facts['scene_clip_summary'], new_fact_node_ids, timestamp_start, timestamp_end)
                    # Save current clip summary memory state
                    self._save_level_graph_and_embeddings(graph=self.clip_summary_memory, embeddings=self.clip_summary_embeddings[:int(
                        clip_id) + 1, :], level=1, before_clip_id=clip_id)
                    # Update highlevel memory state
                    highlevel_memory_embedding = self._build_or_update_highlevel_memory_state(
                        facts['clip_summary'], int(clip_id))                   # Save current highlevel memory state
                    self._save_level_graph_and_embeddings(
                        graph=self.highlevel_memory, embeddings=highlevel_memory_embedding, level=2, before_clip_id=clip_id)
                    # Compose all levels up to current clip
                    self._compose_all_levels(before_clip_id=clip_id)

            stats_delta = self._diff_stats(stats_before, self._snapshot_stats())
            self.per_clip_token_rows.append({
                "clip_id": int(clip_id),
                "qwen_in": stats_delta["qwen"]["prompt_tokens"],
                "qwen_out": stats_delta["qwen"]["completion_tokens"],
                "qwen_calls": stats_delta["qwen"]["num_calls"],
                "gemini_in": stats_delta["gemini"]["prompt_tokens"],
                "gemini_out": stats_delta["gemini"]["completion_tokens"],
                "gemini_calls": stats_delta["gemini"]["num_calls"],
            })

        self._dump_token_usage(video_id)

        if self.two_level_mode:
            self._compose_all_levels()
            return

        self.run_hierarchy(video_id, max_hierarchy_level=max_hierarchy_level)

    def _aggregate_token_usage(self) -> dict:
        """Per-source totals (per video) + average per clip."""
        ts = self.token_stats
        n_clips = len(self.per_clip_token_rows)

        def _per_source(src):
            tin = ts[src]["prompt_tokens"]
            tout = ts[src]["completion_tokens"]
            return {
                "total_in_per_video": tin,
                "total_out_per_video": tout,
                "avg_in_per_clip": (tin / n_clips) if n_clips else 0.0,
                "avg_out_per_clip": (tout / n_clips) if n_clips else 0.0,
                "num_calls": ts[src]["num_calls"],
            }

        return {
            "num_clips": n_clips,
            "qwen": _per_source("qwen"),
            "gemini": _per_source("gemini"),
        }

    def _dump_token_usage(self, video_id: str):
        """Write per-clip CSV and per-video summary JSON next to the graph outputs."""
        per_clip_path = os.path.join(self.graph_out_dir, f"{video_id}_token_usage_per_clip.csv")
        summary_path = os.path.join(self.graph_out_dir, f"{video_id}_token_usage_summary.json")

        fieldnames = ["clip_id", "qwen_in", "qwen_out", "qwen_calls", "gemini_in", "gemini_out", "gemini_calls"]
        with open(per_clip_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.per_clip_token_rows)

        agg = self._aggregate_token_usage()
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)

        q, g = agg["qwen"], agg["gemini"]
        print(
            f"[Tokens][{video_id}] clips={agg['num_clips']} | "
            f"qwen total in/out={q['total_in_per_video']}/{q['total_out_per_video']} "
            f"avg/clip in/out={q['avg_in_per_clip']:.1f}/{q['avg_out_per_clip']:.1f} | "
            f"gemini total in/out={g['total_in_per_video']}/{g['total_out_per_video']} "
            f"avg/clip in/out={g['avg_in_per_clip']:.1f}/{g['avg_out_per_clip']:.1f}"
        )

    def build_memory(self,
                     video_id,
                     metadata,
                     embeddings,
                     doc_mask,
                     max_hierarchy_level: int = 10):

        self._prepare_output_dirs(video_id)

        # Merge metadata from different clips
        merged_metadata = [fact for _, facts in metadata.items()
                           for fact in facts]
        # Add nodes
        self._add_nodes_from_metadata(merged_metadata)
        self.embeddings = np.atleast_2d(embeddings).astype(np.float32)
        self.doc_mask = doc_mask  # None for single-doc; (N,N) for multi-doc

        # Build edges from pairwise similarity for level-0
        sim = self._build_pairwise_similarity()
        self._add_edges_from_similarity(sim)

        # Run hierarchy & compose all levels
        self.run_hierarchy(video_id, max_hierarchy_level=max_hierarchy_level)

# FIXME: this could be moved to a utils file
def load_json(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def load_single_video(dataset: str, video_id: str):

    meta_fp = f'./processed_data/{dataset}/fact_metadata/{video_id}_metadata.json'
    emb_fp =  f'./processed_data/{dataset}/fact_embeddings/{video_id}_embeddings.npy'

    meta = load_json(meta_fp)
    emb_dict = np.load(emb_fp, allow_pickle=True).item()
    fact_emb = emb_dict['facts']
    clip_summary_emb = emb_dict['clip_summary']
    return meta, fact_emb, clip_summary_emb

def single_video_worker(filename,
                      dataset,
                      cam_kwargs,
                      online_mode,
                      max_hierarchy_level):
    try:
        video_id = filename.split(".")[0]
        metadata, fact_embeddings, clip_summary_embeddings = load_single_video(
            dataset, video_id)
        if online_mode:
            cam = CAM(**cam_kwargs)
            cam.build_online_memory(video_id=video_id,
                                    fact_embeddings=fact_embeddings,
                                    clip_summary_embeddings=clip_summary_embeddings,
                                    metadata=metadata,
                                    max_hierarchy_level=max_hierarchy_level)
        else:
            # FIXME: Currently this execution xpath is not considered
            cam = CAM(**cam_kwargs)
            cam.build_memory(video_id=video_id,
                             metadata=metadata,
                             embeddings=fact_embeddings,
                             doc_mask=None,
                             max_hierarchy_level=max_hierarchy_level)
        token_agg = cam._aggregate_token_usage() if hasattr(cam, "_aggregate_token_usage") else None
        return (video_id, True, "ok", token_agg)
    except Exception as e:
        return (filename, False, str(e), None)

def main():
    args = parse_args()

    files = [f for f in os.listdir(args.facts_dir) if f.endswith('.json')]
    files.sort()

    if not files:
        print(f"[Warn] No files matched '*.json' in {args.facts_dir}")
        return

    default_procs = min(10, cpu_count() or 1)
    num_processes = args.num_processes or default_procs
    print(f"[Pool] Using processes: {num_processes}")

    qwen_model = None
    if args.qwen_server and args.qwen:
        print(f"[Error] Qwen server and local Qwen model cannot be used together")
        return
    if args.qwen and args.llm_link:
        print(f"[Init] Loading Qwen model: {args.link_model}")
        qwen_model = APIClient("qwen", None, args.link_model, None, debug=False, download_dir="./models")

    cam_kwargs = dict(
        dataset=args.dataset,
        threshold=args.threshold,
        weight=args.weight,
        sigma=args.sigma,
        top_k=args.k,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        rrf_k=args.rrf_k,
        enable_entity_relation_overlap=args.enable_entity_relation_overlap,
        entity_top_k=args.entity_top_k,
        relation_top_k=args.relation_top_k,
        entity_relation_alpha=args.entity_relation_alpha,
        api_key_path=args.api_key_path,
        link_api_key_path=args.link_api_key_path,
        model=args.model,
        embedding_model=args.embedding_model,
        link_model=args.link_model,
        llm_link=args.llm_link,
        max_cluster_size=args.max_cluster_size,
        summary_field=args.summary_field,
        super_graph_dir=args.super_graph_dir,
        super_embedding_dir=args.super_embedding_dir,
        vector_store_dir=args.vector_store_dir,
        use_qwen=args.qwen or args.qwen_server,
        qwen_model=qwen_model,
        qwen_server=args.qwen_server
    )

    successes = 0
    per_video_aggs = []

    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                single_video_worker,
                filename=filename,
                dataset=args.dataset,
                cam_kwargs=cam_kwargs,
                online_mode=args.online_mode,
                max_hierarchy_level=args.max_hierarchy_level
            ): filename
            for filename in files
        }

        # Process results as they complete
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                book_title, ok, msg, token_agg = future.result()
                if ok:
                    successes += 1
                    print(f"[✓] {book_title}")
                    if token_agg is not None:
                        per_video_aggs.append((book_title, token_agg))
                else:
                    print(f"[✗] {book_title}: {msg}")
            except Exception as e:
                print(f"[✗] {filename}: Exception during execution: {e}")
        print(
            f"[Summary] {successes}/{len(files)} single-document memories completed.")

    if per_video_aggs:
        n_videos = len(per_video_aggs)
        total_clips = sum(a["num_clips"] for _, a in per_video_aggs)

        def _src_totals(src):
            tin = sum(a[src]["total_in_per_video"] for _, a in per_video_aggs)
            tout = sum(a[src]["total_out_per_video"] for _, a in per_video_aggs)
            return tin, tout

        qwen_in, qwen_out = _src_totals("qwen")
        gem_in, gem_out = _src_totals("gemini")

        def _block(tin, tout):
            return {
                "avg_in_per_clip": (tin / total_clips) if total_clips else 0.0,
                "avg_out_per_clip": (tout / total_clips) if total_clips else 0.0,
                "avg_in_per_video": (tin / n_videos) if n_videos else 0.0,
                "avg_out_per_video": (tout / n_videos) if n_videos else 0.0,
                "total_in": tin,
                "total_out": tout,
            }

        dataset_summary = {
            "dataset": args.dataset,
            "num_videos": n_videos,
            "num_clips": total_clips,
            "qwen": _block(qwen_in, qwen_out),
            "gemini": _block(gem_in, gem_out),
        }

        out_dir = os.path.join(args.super_graph_dir, args.dataset)
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, "dataset_token_usage_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(dataset_summary, f, ensure_ascii=False, indent=2)

        q, g = dataset_summary["qwen"], dataset_summary["gemini"]
        print(f"[Dataset Tokens] videos={n_videos} clips={total_clips}")
        print(f"  qwen   avg/clip in/out={q['avg_in_per_clip']:.1f}/{q['avg_out_per_clip']:.1f} | "
              f"avg/video in/out={q['avg_in_per_video']:.1f}/{q['avg_out_per_video']:.1f}")
        print(f"  gemini avg/clip in/out={g['avg_in_per_clip']:.1f}/{g['avg_out_per_clip']:.1f} | "
              f"avg/video in/out={g['avg_in_per_video']:.1f}/{g['avg_out_per_video']:.1f}")
        print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
