import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.models.document_schema import LDU


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_numbers: list[int] = Field(default_factory=list)
    parent_ldu_id: str


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uid: str
    content: str
    content_hash: str
    metadata: ChunkMetadata
    token_count: int

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chunk content must not be empty")
        return value


class ChunkValidator:
    """Validation helper for chunk integrity checks."""

    HASH_PATTERNS = {
        "md5": re.compile(r"^[a-f0-9]{32}$"),
        "sha256": re.compile(r"^[a-f0-9]{64}$"),
    }

    def __init__(self, max_tokens: int, hash_algorithm: Literal["md5", "sha256"] = "sha256") -> None:
        self.max_tokens = max_tokens
        self.hash_algorithm = hash_algorithm

    def validate(self, chunk: Chunk, seen_hashes: set[str]) -> None:
        self._validate_not_empty(chunk)
        self._validate_token_limit(chunk)
        self._validate_hash(chunk, seen_hashes)

    def _validate_not_empty(self, chunk: Chunk) -> None:
        if not chunk.content.strip():
            raise ValueError(f"Chunk {chunk.uid} is empty")

    def _validate_token_limit(self, chunk: Chunk) -> None:
        if chunk.token_count > self.max_tokens:
            raise ValueError(
                f"Chunk {chunk.uid} exceeds token limit: {chunk.token_count} > {self.max_tokens}"
            )

    def _validate_hash(self, chunk: Chunk, seen_hashes: set[str]) -> None:
        pattern = self.HASH_PATTERNS[self.hash_algorithm]
        if not pattern.fullmatch(chunk.content_hash):
            raise ValueError(f"Chunk {chunk.uid} has invalid {self.hash_algorithm} hash format")

        expected_hash = ChunkingEngine.compute_hash(chunk.content, self.hash_algorithm)
        if chunk.content_hash != expected_hash:
            raise ValueError(f"Chunk {chunk.uid} has incorrect content hash")

        if chunk.content_hash in seen_hashes:
            raise ValueError(f"Chunk {chunk.uid} has duplicate content hash")


class ChunkingEngine:
    """Splits LDU content into sentence-aware, overlapping chunks."""

    SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_tokens: int = 500, hash_algorithm: Literal["md5", "sha256"] = "sha256") -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = max(1, int(round(max_tokens * 0.1)))
        self.hash_algorithm = hash_algorithm
        self.validator = ChunkValidator(max_tokens=max_tokens, hash_algorithm=hash_algorithm)

    def chunk_ldu(self, ldu: LDU) -> list[Chunk]:
        sentences = self._split_sentences(ldu.content)
        chunks: list[Chunk] = []
        seen_hashes: set[str] = set()

        sentence_idx = 0
        chunk_index = 0
        while sentence_idx < len(sentences):
            chunk_sentences: list[str] = []
            token_count = 0

            while sentence_idx < len(sentences):
                sentence = sentences[sentence_idx].strip()
                if not sentence:
                    sentence_idx += 1
                    continue

                sentence_tokens = self._count_tokens(sentence)
                if chunk_sentences and (token_count + sentence_tokens) > self.max_tokens:
                    break

                if not chunk_sentences and sentence_tokens > self.max_tokens:
                    sentence = self._trim_to_token_limit(sentence, self.max_tokens)
                    sentence_tokens = self._count_tokens(sentence)

                chunk_sentences.append(sentence)
                token_count += sentence_tokens
                sentence_idx += 1

                if token_count >= self.max_tokens:
                    break

            if not chunk_sentences:
                break

            chunk_content = " ".join(chunk_sentences).strip()
            chunk = Chunk(
                uid=f"{ldu.uid}-chunk-{chunk_index:03d}",
                content=chunk_content,
                content_hash=self.compute_hash(chunk_content, self.hash_algorithm),
                metadata=ChunkMetadata(
                    page_numbers=list(ldu.page_refs),
                    parent_ldu_id=ldu.uid,
                ),
                token_count=self._count_tokens(chunk_content),
            )
            self.validator.validate(chunk, seen_hashes)
            seen_hashes.add(chunk.content_hash)
            chunks.append(chunk)
            chunk_index += 1

            if sentence_idx >= len(sentences):
                break

            sentence_idx = self._rewind_for_overlap(sentences, sentence_idx)

        return chunks

    @staticmethod
    def compute_hash(content: str, algorithm: Literal["md5", "sha256"] = "sha256") -> str:
        encoded = content.encode("utf-8")
        if algorithm == "md5":
            return hashlib.md5(encoded).hexdigest()
        return hashlib.sha256(encoded).hexdigest()

    def _split_sentences(self, content: str) -> list[str]:
        clean_content = (content or "").strip()
        if not clean_content:
            return []
        return [part for part in self.SENTENCE_BOUNDARY_PATTERN.split(clean_content) if part.strip()]

    def _count_tokens(self, text: str) -> int:
        # Lightweight token approximation for chunking constraints.
        return len(text.split())

    def _trim_to_token_limit(self, sentence: str, limit: int) -> str:
        tokens = sentence.split()
        if len(tokens) <= limit:
            return sentence
        return " ".join(tokens[:limit]).strip()

    def _rewind_for_overlap(self, sentences: list[str], next_index: int) -> int:
        target_overlap = self.overlap_tokens
        rewind_index = next_index
        carried_tokens = 0

        while rewind_index > 0 and carried_tokens < target_overlap:
            rewind_index -= 1
            carried_tokens += self._count_tokens(sentences[rewind_index])

        return max(0, rewind_index)
