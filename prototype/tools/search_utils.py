from __future__ import annotations

import re
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from rank_bm25 import BM25Okapi
from prototype.tools.prompts import entity_extraction_template, relation_extraction_template
from prototype.tools.vectorstore import Qdrant

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from prototype.tools.api_client import APIClient

STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "either", "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "because", "if", "when", "where", "how",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "about", "up",
})

_WORD_RE = re.compile(r"[a-z0-9]+")
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```(?:[\w+-]+)?|```")
_SPLIT_RE = re.compile(r"[;\n]+")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")

MAX_RETRIES = 5

@dataclass
class FusedResult:
    """Mimics qdrant ScoredPoint interface so callers can access .payload."""
    id: int
    payload: dict
    score: float = 0.0
    sources: list = field(default_factory=list)


class MultiSearcher:
    """Hybrid dense, BM25, entity, and relation search with RRF fusion."""

    def __init__(
        self,
        collection_name: str,
        path: str,
        embedding_model_dims: int,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        rrf_k: int = 60,
        entity_client: Optional["APIClient"] = None,
        enable_entity_relation_overlap: bool = False,
        entity_top_k: int | None = None,
        relation_top_k: int | None = None,
        entity_relation_alpha: float = 0.5,
    ):
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.entity_client = entity_client
        self.enable_entity_relation_overlap = enable_entity_relation_overlap
        self.entity_top_k = entity_top_k if entity_top_k is not None else sparse_top_k
        self.relation_top_k = relation_top_k if relation_top_k is not None else sparse_top_k
        self.entity_relation_alpha = float(np.clip(entity_relation_alpha, 0.0, 1.0))

        if self.enable_entity_relation_overlap and self.entity_client is None:
            raise ValueError("entity_client must be provided when enable_entity_relation_overlap=True")

        self.qdrant = Qdrant(
            collection_name=collection_name,
            path=path,
            embedding_model_dims=embedding_model_dims,
        )

        self._corpus_tokens: list[list[str]] = []
        self._corpus_ids: list[int] = []
        self._corpus_payloads: list[dict] = []
        self._corpus_entities: list[list[str]] = []
        self._corpus_relations: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase word tokenization with stopword removal."""
        return [
            tok for tok in _WORD_RE.findall(text.lower())
            if tok not in STOPWORDS and len(tok) > 1
        ]

    @staticmethod
    def _strip_model_artifacts(response: str | None) -> str:
        if not isinstance(response, str):
            return ""
        cleaned = _THINK_RE.sub("", response)
        cleaned = _FENCE_RE.sub("", cleaned)
        return cleaned.strip()

    @staticmethod
    def _normalize_entities(entities: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for entity in entities:
            if not isinstance(entity, str):
                continue
            clean = _BULLET_PREFIX_RE.sub("", entity.strip().lower())
            clean = re.sub(r"\s+", " ", clean)
            if not clean or clean in seen:
                continue
            seen.add(clean)
            normalized.append(clean)
        return normalized

    @staticmethod
    def _normalize_relation_triplets(relations: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for relation in relations:
            if not isinstance(relation, str):
                continue
            clean = _BULLET_PREFIX_RE.sub("", relation.strip().lower())
            parts = [re.sub(r"\s+", " ", part.strip()) for part in clean.split("|")]
            if len(parts) != 3 or any(not part for part in parts):
                continue
            triplet = "|".join(parts)
            if triplet in seen:
                continue
            seen.add(triplet)
            normalized.append(triplet)
        return normalized

    def _parse_entity_response(self, response: str | None) -> list[str]:
        cleaned = self._strip_model_artifacts(response)
        if not cleaned:
            return []
        return self._normalize_entities([chunk for chunk in _SPLIT_RE.split(cleaned) if chunk.strip()])

    def _parse_relation_response(self, response: str | None) -> list[str]:
        cleaned = self._strip_model_artifacts(response)
        if not cleaned:
            return []
        return self._normalize_relation_triplets([chunk for chunk in _SPLIT_RE.split(cleaned) if chunk.strip()])

    def _extract_entities(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        if self.entity_client is None:
            raise ValueError("entity_client is required for entity extraction")

        prompt = entity_extraction_template.format(input_chunk=text)
        for attempt in range(MAX_RETRIES):
            try:
                response = self.entity_client.obtain_response(prompt=prompt, max_tokens=256, temperature=0.0)
                if response is not None:
                    break
            except Exception as exc:
                logger.warning("Entity extraction failed for text %r: %s", text[:120], exc)
                if attempt == MAX_RETRIES - 1:
                    print(f"Return [] since {exc}")
                    return []
        return self._parse_entity_response(response)

    def _extract_relations(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        if self.entity_client is None:
            raise ValueError("entity_client is required for relation extraction")

        prompt = relation_extraction_template.format(input_chunk=text)
        for attempt in range(MAX_RETRIES):
            try:
                response = self.entity_client.obtain_response(prompt=prompt, max_tokens=256, temperature=0.0)
                if response is not None:
                    break
            except Exception as exc:
                logger.warning("Relation extraction failed for text %r: %s", text[:120], exc)
                if attempt == MAX_RETRIES - 1:
                    print(f"Return [] since {exc}")
                    return []
        return self._parse_relation_response(response)

    @staticmethod
    def _overlap_score(query_items: list[str], doc_items: list[str]) -> float:
        if not query_items or not doc_items:
            return 0.0
        overlap = set(query_items) & set(doc_items)
        if not overlap:
            return 0.0
        return float(len(overlap)) / float(len(set(query_items)))

    def insert(self, vectors: list, payloads: list = None, ids: list = None):
        """Insert dense vectors into Qdrant and update the in-memory search indexes.

        Signature matches Qdrant.insert() so it's a drop-in replacement.
        """
        self.qdrant.insert(vectors, payloads, ids)

        for idx in range(len(vectors)):
            payload = payloads[idx] if payloads else {}
            cached_payload = dict(payload) if isinstance(payload, dict) else {}
            point_id = ids[idx] if ids is not None else idx
            raw_text = cached_payload.get("raw_fact_text", "")
            self._corpus_tokens.append(self._tokenize(raw_text))
            self._corpus_ids.append(point_id)
            if self.enable_entity_relation_overlap:
                cached_payload["entities"] = self._extract_entities(raw_text)
                cached_payload["entity_relations"] = self._extract_relations(raw_text)
            self._corpus_entities.append(list(cached_payload.get("entities", [])))
            self._corpus_relations.append(list(cached_payload.get("entity_relations", [])))
            self._corpus_payloads.append(cached_payload)

        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query=None, vectors=None, limit: int = 5, filters=None):
        """Dense-only search, backward-compatible with Qdrant.search()."""
        return self.qdrant.search(query=query, vectors=vectors, limit=limit, filters=filters)

    def _search_dense(self, query_vector: list, limit: int, filters=None) -> list:
        return self.qdrant.search(query=None, vectors=query_vector, limit=limit, filters=filters)

    def _search_bm25(self, query_text: str, limit: int, payload_filter: Callable[[dict], bool] | None = None) -> list[tuple[int, dict, float]]:
        """Return top-k BM25 results as (point_id, payload, score) tuples."""
        if self._bm25 is None or not self._corpus_tokens:
            return []
        tokenized_query = self._tokenize(query_text)
        if not tokenized_query:
            return []
        scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1]
        hits: list[tuple[int, dict, float]] = []
        for i in ranked_indices:
            if scores[i] <= 0:
                break
            payload = self._corpus_payloads[i]
            if payload_filter is not None and not payload_filter(payload):
                continue
            hits.append((self._corpus_ids[i], payload, float(scores[i])))
            if len(hits) >= limit:
                break
        return hits

    def _search_entity_overlap(self, query_text: str, limit: int, payload_filter: Callable[[dict], bool] | None = None) -> list[tuple[int, dict, float]]:
        if not self.enable_entity_relation_overlap:
            return []
        query_entities = self._extract_entities(query_text)
        if not query_entities:
            return []

        hits: list[tuple[int, dict, float]] = []
        for pid, payload, doc_entities in zip(self._corpus_ids, self._corpus_payloads, self._corpus_entities):
            if payload_filter is not None and not payload_filter(payload):
                continue
            score = self._overlap_score(query_entities, doc_entities)
            if score <= 0:
                continue
            hits.append((pid, payload, score))

        hits.sort(key=lambda item: item[2], reverse=True)
        return hits[:limit]

    def _search_relation_overlap(self, query_text: str, limit: int, payload_filter: Callable[[dict], bool] | None = None) -> list[tuple[int, dict, float]]:
        if not self.enable_entity_relation_overlap:
            return []
        query_relations = self._extract_relations(query_text)
        if not query_relations:
            return []

        hits: list[tuple[int, dict, float]] = []
        for pid, payload, doc_relations in zip(self._corpus_ids, self._corpus_payloads, self._corpus_relations):
            if payload_filter is not None and not payload_filter(payload):
                continue
            score = self._overlap_score(query_relations, doc_relations)
            if score <= 0:
                continue
            hits.append((pid, payload, score))

        hits.sort(key=lambda item: item[2], reverse=True)
        return hits[:limit]

    def _rrf_fuse(self, ranked_lists: list[list[tuple]], limit: int) -> list[FusedResult]:
        """Reciprocal Rank Fusion across multiple ranked result lists.

        Each element of ranked_lists is a list of (point_id, payload, source_tag) tuples
        in descending relevance order.
        """
        scores: dict[int, float] = {}
        payloads: dict[int, dict] = {}
        sources: dict[int, list[str]] = {}

        for ranked in ranked_lists:
            for rank, (pid, payload, tag) in enumerate(ranked):
                scores[pid] = scores.get(pid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                payloads.setdefault(pid, payload)
                sources.setdefault(pid, []).append(tag)

        sorted_ids = sorted(scores, key=lambda pid: scores[pid], reverse=True)[:limit]
        return [
            FusedResult(
                id=pid,
                payload=payloads[pid],
                score=scores[pid],
                sources=sources[pid],
            )
            for pid in sorted_ids
        ]

    def search_multi(
        self,
        query_vector: list,
        query_text: str,
        limit: int,
        filters=None,
        payload_filter: Callable[[dict], bool] | None = None,
        use_entity_relation_overlap: bool | None = None,
    ) -> list[FusedResult]:
        """Run dense, BM25, and optional entity/relation search and fuse with RRF."""
        dense_hits = self._search_dense(query_vector, self.dense_top_k, filters=filters)
        bm25_hits = self._search_bm25(query_text, self.sparse_top_k, payload_filter=payload_filter)

        entity_relation_enabled = self.enable_entity_relation_overlap if use_entity_relation_overlap is None else use_entity_relation_overlap
        entity_hits: list[tuple[int, dict, float]] = []
        relation_hits: list[tuple[int, dict, float]] = []
        if entity_relation_enabled:
            entity_hits = self._search_entity_overlap(query_text, self.entity_top_k, payload_filter=payload_filter)
            relation_hits = self._search_relation_overlap(query_text, self.relation_top_k, payload_filter=payload_filter)

        dense_ranked = [(h.id, h.payload, "dense") for h in dense_hits]
        bm25_ranked = [(pid, payload, "bm25") for pid, payload, _score in bm25_hits]
        entity_ranked = [(pid, payload, "entity") for pid, payload, _score in entity_hits]
        relation_ranked = [(pid, payload, "relation") for pid, payload, _score in relation_hits]

        ranked_lists = [dense_ranked]
        if bm25_ranked:
            ranked_lists.append(bm25_ranked)
        if entity_ranked:
            ranked_lists.append(entity_ranked)
        if relation_ranked:
            ranked_lists.append(relation_ranked)

        return self._rrf_fuse(ranked_lists, limit)

    def close(self):
        self.qdrant.close()
