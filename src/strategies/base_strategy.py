from abc import ABC, abstractmethod

from src.models.document_schema import DocumentProfile, ExtractedDocument


class BaseStrategy(ABC):
	"""Common interface for all extraction strategies (A, B, and C)."""

	@abstractmethod
	def extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
		"""Extract structured content from a PDF using the provided profile."""
