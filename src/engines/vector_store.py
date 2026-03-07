from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.models.document_schema import Chunk


class VectorStoreManager:
    """Manage local Chroma storage and retrieval for chunked documents."""

    def __init__(
        self,
        db_path: str = "data/vector_db",
        collection_name: str | None = None,
        source_filename: str | None = None,
    ) -> None:
        del db_path  # Enforce fixed persistent location for reliability.
        self.db_path = os.path.join(os.getcwd(), ".refinery", "vector_db")
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(is_persistent=True),
        )
        self.embedding_function = DefaultEmbeddingFunction()
        self.collection_name = self._resolve_collection_name(
            collection_name=collection_name,
            source_filename=source_filename,
        )
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def _resolve_collection_name(self, collection_name: str | None, source_filename: str | None) -> str:
        if collection_name:
            return collection_name

        if source_filename:
            return self._sanitize_collection_name(source_filename)

        return "document_chunks"

    def _sanitize_collection_name(self, source_filename: str) -> str:
        stem = Path(source_filename).stem.strip().lower()
        sanitized = re.sub(r"[^a-z0-9_-]+", "_", stem)
        sanitized = sanitized.strip("_-")
        if not sanitized:
            sanitized = "document_chunks"

        # Chroma collection names must be reasonably short and stable.
        if len(sanitized) > 60:
            sanitized = sanitized[:60].rstrip("_-")
        return sanitized

    def _ensure_collection(self) -> Collection:
        """Always return a valid collection, creating it when necessary."""
        try:
            return self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
        except Exception:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        self.collection = self._ensure_collection()

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
                content_hash = str(raw_metadata.get("content_hash", chunk.content_hash))
                bbox = raw_metadata.get("bbox", "")
            else:
                page_numbers = getattr(raw_metadata, "page_numbers", [])
                parent_ldu_id = str(getattr(raw_metadata, "parent_ldu_id", ""))
                unit_type = str(getattr(raw_metadata, "unit_type", ""))
                title = str(getattr(raw_metadata, "title", ""))
                filename = str(getattr(raw_metadata, "filename", ""))
                content_hash = str(getattr(raw_metadata, "content_hash", chunk.content_hash))
                bbox = getattr(raw_metadata, "bbox", "")

            if not isinstance(page_numbers, list):
                page_numbers = [page_numbers] if page_numbers else []

            normalized_pages: list[int] = []
            for page in page_numbers:
                try:
                    normalized_pages.append(int(page))
                except (TypeError, ValueError):
                    continue

            ids.append(chunk.uid)
            documents.append(chunk.content)
            metadatas.append(
                {
                    "uid": chunk.uid,
                    "content_hash": content_hash,
                    "page_numbers": ",".join(str(page) for page in normalized_pages),
                    "page_number": normalized_pages[0] if normalized_pages else 1,
                    "parent_ldu_id": parent_ldu_id,
                    "unit_type": unit_type,
                    "title": title,
                    "filename": filename,
                    "bbox": str(bbox),
                }
            )

        print(f"DEBUG: Writing to [{self.db_path}]")
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, text: str, n_results: int = 5, page_filter: List[int] | None = None) -> list[dict[str, Any]]:
        if not text.strip():
            return []

        self.collection = self._ensure_collection()

        where_clause: dict[str, Any] | None = None
        filter_pages: list[int] = []
        if page_filter:
            for page in page_filter:
                try:
                    filter_pages.append(int(page))
                except (TypeError, ValueError):
                    continue

            # Primary fast path: use Chroma metadata filtering by canonical `page_number`.
            if filter_pages:
                where_clause = {"page_number": {"$in": filter_pages}}

        expanded_k = max(n_results, n_results * 5) if page_filter else n_results
        try:
            raw_results = self.collection.query(
                query_texts=[text],
                n_results=expanded_k,
                include=["documents", "metadatas", "distances"],
                where=where_clause,
            )
        except Exception:
            # Resilient fallback in case metadata operators differ by Chroma version.
            raw_results = self.collection.query(
                query_texts=[text],
                n_results=expanded_k,
                include=["documents", "metadatas", "distances"],
            )

        ids = (raw_results.get("ids") or [[]])[0]
        documents = (raw_results.get("documents") or [[]])[0]
        metadatas = (raw_results.get("metadatas") or [[]])[0]
        distances = (raw_results.get("distances") or [[]])[0]

        filter_set = set(filter_pages)
        matches: list[dict[str, Any]] = []

        for idx, doc_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            doc_text = documents[idx] if idx < len(documents) else ""
            distance = distances[idx] if idx < len(distances) else None

            # If we have a filter, only skip if there is absolutely no overlap.
            if filter_set:
                candidate_pages = self._parse_page_numbers(metadata.get('page_numbers', ''))
                # Fallback: if metadata is missing page_numbers, check the singular page_number.
                if not candidate_pages and 'page_number' in metadata:
                    try:
                        candidate_pages = {int(metadata['page_number'])}
                    except (TypeError, ValueError):
                        candidate_pages = set()

                if filter_set and candidate_pages and not filter_set.intersection(candidate_pages):
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
