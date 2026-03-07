from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.models.document_schema import Chunk


class VectorStoreManager:
    """Manage local Chroma storage and retrieval for chunked documents."""

    def __init__(
        self,
        db_path: str = "data/vector_db",
        collection_name: str = "document_chunks",
    ) -> None:
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = DefaultEmbeddingFunction()
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            raw_metadata = chunk.metadata or {}
            if isinstance(raw_metadata, dict):
                page_numbers = raw_metadata.get("page_numbers", [])
                parent_ldu_id = str(raw_metadata.get("parent_ldu_id", ""))
                unit_type = str(raw_metadata.get("unit_type", ""))
                title = str(raw_metadata.get("title", ""))
                filename = str(raw_metadata.get("filename", ""))
            else:
                page_numbers = getattr(raw_metadata, "page_numbers", [])
                parent_ldu_id = str(getattr(raw_metadata, "parent_ldu_id", ""))
                unit_type = str(getattr(raw_metadata, "unit_type", ""))
                title = str(getattr(raw_metadata, "title", ""))
                filename = str(getattr(raw_metadata, "filename", ""))

            if not isinstance(page_numbers, list):
                page_numbers = [page_numbers] if page_numbers else []

            ids.append(chunk.uid)
            documents.append(chunk.content)
            metadatas.append(
                {
                    "uid": chunk.uid,
                    "page_numbers": ",".join(str(page) for page in page_numbers),
                    "parent_ldu_id": parent_ldu_id,
                    "unit_type": unit_type,
                    "title": title,
                    "filename": filename,
                }
            )

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, text: str, n_results: int = 3, page_filter: list[int] | None = None) -> list[dict[str, Any]]:
        if not text.strip():
            return []

        expanded_k = max(n_results, n_results * 5) if page_filter else n_results
        raw_results = self.collection.query(
            query_texts=[text],
            n_results=expanded_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (raw_results.get("ids") or [[]])[0]
        documents = (raw_results.get("documents") or [[]])[0]
        metadatas = (raw_results.get("metadatas") or [[]])[0]
        distances = (raw_results.get("distances") or [[]])[0]

        filter_set = set(page_filter or [])
        matches: list[dict[str, Any]] = []

        for idx, doc_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            doc_text = documents[idx] if idx < len(documents) else ""
            distance = distances[idx] if idx < len(distances) else None

            if filter_set:
                candidate_pages = self._parse_page_numbers(metadata.get("page_numbers", ""))
                if not filter_set.intersection(candidate_pages):
                    continue

            matches.append(
                {
                    "id": doc_id,
                    "document": doc_text,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

            if len(matches) >= n_results:
                break

        return matches

    def _parse_page_numbers(self, raw_value: Any) -> set[int]:
        if raw_value is None:
            return set()

        text_value = str(raw_value).strip()
        if not text_value:
            return set()

        pages: set[int] = set()
        for part in text_value.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                pages.add(int(part))
            except ValueError:
                continue

        return pages
