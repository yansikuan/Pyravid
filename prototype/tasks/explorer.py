from errno import ESTALE
from typing import List, Set, Tuple
import numpy as np
import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity

from prototype.tools.api_client import APIClient
from prototype.tools.utils import load_memory_graph


class Explorer:
    """
    Base class for all explorers.
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
        graph: nx.Graph = None,
        selection_model: str = "gemini-2.0-flash",
        answer_ip_address: str = None,
        selection_ip_address: str = None
    ):
        self.dataset = dataset
        self.video_id = video_id
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.two_level_mode = two_level_mode
        self.graph_path = f"./{super_graph_dir}/{dataset}/{video_id}/{video_id}_graph_level_all.gexf"
        self.embedding_path = f"./{super_embedding_dir}/{dataset}/{video_id}/{video_id}_embedding_level_all.npy"
        if graph:
            self.memory_graph = graph
        else:
            self.memory_graph = load_memory_graph(self.graph_path)
        self.memory_embedding = np.load(self.embedding_path).astype(np.float32)
        if self.two_level_mode:
            self.memory_embedding = self.memory_embedding[:-1, :]

        self.client = APIClient("openai", api_key_path, model, embedding_model)
        self.answer_client = APIClient("qwen-server-answer-model", None, answer_model, None, debug=True, ip_address=answer_ip_address, think_mode=False)
        self.selection_client = APIClient("qwen-server-selection-model", None, selection_model, None, debug=True, ip_address=selection_ip_address, think_mode=False)
        self.multimodal = multimodal

    def _embed_question(self, question: str) -> np.ndarray:
        emb = self.client.obtain_embedding(question)
        return np.array(emb, dtype=np.float32)

    def _initial_topk(self, q_emb: np.ndarray) -> List[int]:
        scores = cosine_similarity(q_emb[np.newaxis, :], self.memory_embedding)[0]
        idx = np.argsort(-scores)[: self.top_k]
        print(f"top-{self.top_k} idx: {idx.tolist()}")
        return idx.tolist()

    def _node_level(self, node_id: int) -> int:
        n = int(node_id)
        if n not in self.memory_graph.nodes:
            print(f"[Warning] Node {n} not in memory_graph, skip.")
            return ""
        return self.memory_graph.nodes[n].get("level", "")

    def _node_image(self, node_id: int) -> str:
        n = int(node_id)
        if n not in self.memory_graph.nodes:
            print(f"[Warning] Node {n} not in memory_graph, skip.")
            return ""
        image_path = self.memory_graph.nodes[n].get("image_path", "")
        try:
            with open(image_path, "r") as f:
                image_b64 = f.read()
                # FIXME: Hotfix, however the video extraction process should be implemented accrodingly
                return image_b64.split(',')[1]
        except Exception as e:
            return ""

    def _node_timestamp(self, node_id: int) -> str:
        n = int(node_id)
        if n not in self.memory_graph.nodes:
            print(f"[Warning] Node {n} not in memory_graph, skip.")
            return {}
        memory_node = self.memory_graph.nodes[n]
        timestamp_dict = {}
        if "timestamp" in memory_node:
            timestamp_dict["timestamp"] = memory_node["timestamp"]
        else:
            timestamp_dict["timestamp_start"] = memory_node["timestamp_start"]
            timestamp_dict["timestamp_end"] = memory_node["timestamp_end"]
        return timestamp_dict

    def _node_asr_periods(self, node_id: int) -> List[str]:
        n = int(node_id)
        if n not in self.memory_graph.nodes:
            print(f"[Warning] Node {n} not in memory_graph, skip.")
            return []
        asr_periods = self.memory_graph.nodes[n].get("asr_periods", [])
        return [asr.get("text") for asr in asr_periods]

    def _merge_level0_chunks(self, node_list: List[Tuple], fact_node_images: Set[int]):
        if not node_list:
            return [], []
        # Sorted by node_id so that all level-0 facts are placed together at beginning of the list
        node_list = sorted(node_list, key=lambda n: n[0])
        txt_result = []
        img_result = []

        def is_level0(nid: int) -> bool:
            return self.memory_graph.nodes[nid].get("level", 0) == 0 if nid in self.memory_graph.nodes else False

        for node_entry in node_list:
            nid, (txt, img) = node_entry
            # Return if current node is not level-0 anymore
            # FIXME: In future may be we could also fetch the facts contained in higher level nodes
            if nid not in self.memory_graph.nodes:
                print(f"[Warning] Node {nid} not in memory_graph, skip.")
                continue
            # if not is_level0(nid):
            #     txt_result.append(f"[LEVEL-{self._node_level(nid)} summary node with community node ids {self.memory_graph.nodes[nid].get('community', [])}] {txt}")
            # else:
            #     txt_result.append(f"[LEVEL-{self._node_level(nid)} fact node] {txt}")
            # if img != "" and nid in fact_node_images:
            #     img_result.append(img)
            txt_result.append(txt)
            if img != "" and nid in fact_node_images:
                img_result.append(img)
        return txt_result, img_result

    def _merge_level0_chunks_with_nid(self, node_list: List[Tuple], fact_node_images: Set[int]):
        if not node_list:
            return [], []
        # Sorted by node_id so that all level-0 facts are placed together at beginning of the list
        node_list = sorted(node_list, key=lambda n: n[0])
        txt_result = []
        img_result = []
        nid_img_mapping = []

        for node_entry in node_list:
            nid, (txt, img) = node_entry
            # Return if current node is not level-0 anymore
            # FIXME: In future may be we could also fetch the facts contained in higher level nodes
            if nid not in self.memory_graph.nodes:
                print(f"[Warning] Node {nid} not in memory_graph, skip.")
                txt = ""
                img = ""

            txt_result.append(txt)
            if img != "" and nid in fact_node_images:
                img_result.append(img)
                nid_img_mapping.append(nid)
        return txt_result, img_result, nid_img_mapping

    def _merge_level0_chunks_wo_img(self, node_list: List[Tuple]):
        if not node_list:
            return [], []
        # Sorted by node_id so that all level-0 facts are placed together at beginning of the list
        node_list = sorted(node_list, key=lambda n: n[0])
        txt_result = []

        for node_entry in node_list:
            nid, txt = node_entry
            # Return if current node is not level-0 anymore
            # FIXME: In future may be we could also fetch the facts contained in higher level nodes
            if nid not in self.memory_graph.nodes:
                print(f"[Warning] Node {nid} not in memory_graph, skip.")
                continue

            txt_result.append(txt)
        return txt_result

    def _node_text(self, node_id: int) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    def _llm_selection(self, question: str, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def run(self, question: str, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
