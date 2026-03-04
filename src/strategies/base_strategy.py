from abc import ABC, abstractmethod
from typing import Any

from src.models.document_schema import DocumentProfile, NormalizedOutput
from src.utils.config_loader import get_chunking_config


class BaseStrategy(ABC):
	"""Common interface for all extraction strategies (A, B, and C)."""

	def __init__(self, config: dict[str, Any] | None = None) -> None:
		self.config = config or {}
		chunking = get_chunking_config(self.config)
		self.max_chunk_size = int(chunking.get("max_chunk_size", 1000))
		overlap = int(chunking.get("overlap_size", 200))
		self.overlap_size = max(0, min(overlap, self.max_chunk_size - 1)) if self.max_chunk_size > 1 else 0

	def chunk_text(self, text: str) -> list[str]:
		content = (text or "").strip()
		if not content:
			return []

		if len(content) <= self.max_chunk_size:
			return [content]

		chunks: list[str] = []
		start = 0
		length = len(content)

		while start < length:
			end = min(length, start + self.max_chunk_size)
			chunk = content[start:end].strip()
			if chunk:
				chunks.append(chunk)

			if end >= length:
				break

			start = max(end - self.overlap_size, start + 1)

		return chunks

	@abstractmethod
	def extract(self, pdf_path: str, profile: DocumentProfile) -> NormalizedOutput:
		"""Extract structured content from a PDF using the provided profile."""
