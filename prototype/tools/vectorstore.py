from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any

import tiktoken

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os
import shutil

encoding = tiktoken.get_encoding('cl100k_base')
logger = logging.getLogger(__name__)
logger.disabled = True

# Vectore Store (Refer to https://github.com/mem0ai/mem0/tree/main)
class VectorStore(ABC):

    @abstractmethod
    def create_col(self, name, vector_size, distance):
        """Create a new collection."""
        pass

    @abstractmethod
    def list_cols(self):
        """List all collections."""
        pass

    @abstractmethod
    def insert(self, vectors, payloads=None, ids=None):
        """Insert vectors into the collection."""
        pass

    @abstractmethod
    def search(self, query, k=10, filter=None):
        """Search for vectors in the collection."""
        pass

class Qdrant(VectorStore):
    def __init__(
        self,
        collection_name: str,
        embedding_model_dims: int,
        client: QdrantClient = None,
        host: str = None,
        port: int = None,
        path: str = None,
        url: str = None,
        api_key: str = None,
        on_disk: bool = False,
        snapshot_dir: str = None,
        embedding_client: Any = None,
    ):
        """
        Initialize the Qdrant vector store.

        Args:
            collection_name (str): Name of the collection.
            embedding_model_dims (int): Dimensions of the embedding model.
            client (QdrantClient, optional): Existing Qdrant client instance. Defaults to None.
            host (str, optional): Host address for Qdrant server. Defaults to None.
            port (int, optional): Port for Qdrant server. Defaults to None.
            path (str, optional): Path for local Qdrant database. Defaults to None.
            url (str, optional): Full URL for Qdrant server. Defaults to None.
            api_key (str, optional): API key for Qdrant server. Defaults to None.
            on_disk (bool, optional): Enables persistent storage. Defaults to False.
            snapshot_dir (str, optional): Directory for local database snapshots.
            embedding_client (optional): Client exposing ``obtain_embedding``.
                It is created lazily from the project OpenAI configuration when
                character or scene records are inserted.
        """
        if client:
            self.client = client
            self.is_local = False
        else:
            params = {}
            if api_key:
                params["api_key"] = api_key
            if url:
                params["url"] = url
            if host and port:
                params["host"] = host
                params["port"] = port

            if not params:
                params["path"] = path
                self.is_local = True
                if not on_disk:
                    if os.path.exists(path) and os.path.isdir(path):
                        shutil.rmtree(path)
            else:
                self.is_local = False

            self.client = QdrantClient(**params)

        self.collection_name = collection_name
        self.path = path
        self.embedding_model_dims = embedding_model_dims
        self.on_disk = on_disk
        self.snapshot_dir = snapshot_dir
        self.embedding_client = embedding_client
        self.create_col(embedding_model_dims, on_disk)

    def create_col(self, vector_size: int, on_disk: bool, distance: Distance = Distance.COSINE):
        """
        Create a new collection.

        Args:
            vector_size (int): Size of the vectors to be stored.
            on_disk (bool): Enables persistent storage.
            distance (Distance, optional): Distance metric for vector similarity. Defaults to Distance.COSINE.
        """
        # Skip creating collection if already exists
        response = self.list_cols()
        for collection in response.collections:
            if collection.name == self.collection_name:
                logger.debug(f"Collection {self.collection_name} already exists. Skipping creation.")
                self._create_filter_indexes()
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance, on_disk=on_disk),
        )
        self._create_filter_indexes()

    def list_cols(self) -> list:
        """
        List all collections.

        Returns:
            list: List of collection names.
        """
        return self.client.get_collections()

    def insert(self, vectors: list, payloads: list = None, ids: list = None):
        """
        Insert vectors into a collection.

        Args:
            vectors (list): List of vectors to insert.
            payloads (list, optional): List of payloads corresponding to vectors. Defaults to None.
            ids (list, optional): List of IDs corresponding to vectors. Defaults to None.
        """
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        points = [
            PointStruct(
                id=idx if ids is None else ids[idx],
                vector=vector,
                payload=payloads[idx] if payloads else {},
            )
            for idx, vector in enumerate(vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, vectors: list, limit: int = 5, filters: dict = None) -> list:
        """
        Search for similar vectors.

        Args:
            query (str): Query. If not None, search by id.
            vectors (list): Query vector. If not None, search by vector.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: Search results.
        """
        if query is not None and vectors is not None:
            raise ValueError("Only one of query or vectors should be provided")
        if query is None and vectors is None:
            raise ValueError("Either query or vectors should be provided")
        if query is not None:
            hit = self.client.retrieve(collection_name=self.collection_name, ids=[query])
            return hit
        else:
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=vectors,
                query_filter=filters,
                limit=limit,
            )
            return hits.points

    def _create_filter_indexes(self):
        """Create indexes for commonly used filter fields to enable filtering."""
        # Only create payload indexes for remote Qdrant servers
        if self.is_local:
            logger.debug("Skipping payload index creation for local Qdrant (not supported)")
            return

        common_fields = ["user_id", "agent_id", "run_id", "actor_id"]

        for field in common_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword"
                )
                logger.info(f"Created index for {field} in collection {self.collection_name}")
            except Exception as e:
                logger.debug(f"Index for {field} might already exist: {e}")

    def _get_embedding_client(self):
        """Return the injected client or lazily create the project default."""
        if self.embedding_client is None:
            from prototype.tools.api_client import APIClient

            project_root = Path(__file__).resolve().parents[2]
            key_path = os.getenv(
                "PYRAVID_OPENAI_KEY_PATH",
                str(project_root / "config" / "openai_key.txt"),
            )
            self.embedding_client = APIClient(
                api="openai",
                key_path=key_path,
                model="",
                embedding_model=os.getenv(
                    "PYRAVID_EMBEDDING_MODEL",
                    "text-embedding-3-large",
                ),
            )
        return self.embedding_client

    def _embed_or_dummy(self, text: str) -> list[float]:
        source_text = text.strip() if isinstance(text, str) else ""
        vector = list(
            self._get_embedding_client().obtain_embedding(source_text or "N/A")
        )
        if len(vector) != self.embedding_model_dims:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.embedding_model_dims}, got {len(vector)}"
            )
        return vector

    @staticmethod
    def _as_payload_list(value: Any) -> list:
        """Normalize collection-like values to Qdrant JSON-compatible lists."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (tuple, set)):
            return list(value)
        return [value]

    @staticmethod
    def _profile_text(person_json: dict) -> str:
        profile = person_json.get("character_profile", "")
        if isinstance(profile, dict):
            profile = profile.get("description", "")
        return str(profile or "")

    def insert_person(self, person_json: dict, point_id: str):
        """Embed and upsert one character profile."""
        point = PointStruct(
            id=point_id,
            vector=self._embed_or_dummy(self._profile_text(person_json)),
            payload={
                "person_id": person_json.get("person_id", point_id),
                "clips": self._as_payload_list(person_json.get("clips")),
                "facts": self._as_payload_list(person_json.get("facts")),
                "character_profile": person_json.get("character_profile", {}),
            },
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def insert_scene(self, scene_json: dict, point_id: str):
        """Embed and upsert one scene record."""
        description = str(scene_json.get("scene_description", "") or "")
        point = PointStruct(
            id=point_id,
            vector=self._embed_or_dummy(description),
            payload={
                "clips": self._as_payload_list(scene_json.get("clips")),
                "facts": self._as_payload_list(scene_json.get("facts")),
                "characters": self._as_payload_list(scene_json.get("characters")),
                "scene_description": description,
            },
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def create_snapshot(self, clip_id: str) -> str:
        """Copy a local Qdrant database into a clip-specific snapshot."""
        if not self.is_local or not self.path:
            raise RuntimeError("Snapshots require a local Qdrant path")
        if not self.snapshot_dir:
            raise ValueError("snapshot_dir is required to create snapshots")

        source_path = os.path.abspath(self.path)
        if not os.path.isdir(source_path):
            raise FileNotFoundError(f"Qdrant path not found: {source_path}")

        os.makedirs(self.snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(
            self.snapshot_dir,
            f"clip_{clip_id}_snapshot",
        )
        if os.path.exists(snapshot_path):
            shutil.rmtree(snapshot_path)
        shutil.copytree(source_path, snapshot_path)
        print(f"  📸 Created local snapshot: {snapshot_path}")
        return snapshot_path

    def close(self):
        """Close the Qdrant client."""
        self.client.close()
