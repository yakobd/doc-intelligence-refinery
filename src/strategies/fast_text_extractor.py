import logging
from pathlib import Path

import pdfplumber

from src.models.document_schema import (
	DocumentProfile,
	ExtractedDocument,
	ExtractedPage,
	Table,
)
from src.strategies.base_strategy import BaseStrategy


logger = logging.getLogger(__name__)


class StrategyA(BaseStrategy):
	"""Fast text-first strategy for native digital and simple-layout PDFs."""

	def extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
		pages: list[ExtractedPage] = []
		pdf_name = Path(pdf_path).name
		doc_id = Path(pdf_path).stem

		with pdfplumber.open(pdf_path) as pdf:
			for page_index, page in enumerate(pdf.pages, start=1):
				text = page.extract_text() or ""
				char_count = len(text)
				image_count = len(page.images or [])
				extraction_confidence = self._calculate_confidence(
					char_count=char_count,
					image_count=image_count,
				)

				if extraction_confidence < 0.5:
					logger.warning(
						"Low extraction confidence on page %s of %s (char_count=%s, image_count=%s, confidence=%.2f)",
						page_index,
						pdf_name,
						char_count,
						image_count,
						extraction_confidence,
					)

				tables: list[Table] = []

				for detected_table in page.find_tables():
					raw_rows = detected_table.extract() or []
					if not raw_rows:
						continue

					header_row = raw_rows[0] or []
					body_rows = raw_rows[1:] if len(raw_rows) > 1 else []

					headers = [self._safe_cell(cell) for cell in header_row]
					rows = [
						[self._safe_cell(cell) for cell in row]
						for row in body_rows
					]

					tables.append(
						Table(
							headers=headers,
							rows=rows,
							title=None,
						)
					)

				pages.append(
					ExtractedPage(
						page_number=page_index,
						text=text,
						tables=tables,
						extraction_confidence=extraction_confidence,
					)
				)

		return ExtractedDocument(
			filename=pdf_name,
			doc_id=doc_id,
			profile=profile,
			pages=pages,
			strategy_used="Strategy A",
		)

	def _safe_cell(self, cell: object) -> str:
		if cell is None:
			return ""
		return str(cell).strip()

	def _calculate_confidence(self, char_count: int, image_count: int) -> float:
		if char_count > 500 and image_count == 0:
			return 1.0
		if char_count < 50 and image_count > 0:
			return 0.1
		if 50 <= char_count <= 500:
			return char_count / 500
		return 1.0
