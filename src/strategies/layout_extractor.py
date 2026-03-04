from pathlib import Path

from docling.document_converter import DocumentConverter

from src.models.document_schema import (
	DocumentProfile,
	ExtractedDocument,
	ExtractedPage,
	Table,
)
from src.strategies.base_strategy import BaseStrategy


class StrategyB(BaseStrategy):
	"""Layout-aware extraction strategy backed by Docling."""

	def extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
		pdf_name = Path(pdf_path).name
		doc_id = Path(pdf_path).stem

		converter = DocumentConverter()
		result = converter.convert(pdf_path)
		document = result.document

		markdown = document.export_to_markdown() if hasattr(document, "export_to_markdown") else ""
		tables = [self._map_docling_table(table_obj) for table_obj in getattr(document, "tables", [])]

		pages = self._build_pages(markdown=markdown, tables=tables)

		return ExtractedDocument(
			filename=pdf_name,
			doc_id=doc_id,
			profile=profile,
			pages=pages,
			strategy_used="Strategy B",
		)

	def _build_pages(self, markdown: str, tables: list[Table]) -> list[ExtractedPage]:
		"""Build page payloads; keep full markdown in page 1 and attach mapped tables."""
		text = markdown or ""
		return [
			ExtractedPage(
				page_number=1,
				text=text,
				tables=tables,
				extraction_confidence=0.9,
			)
		]

	def _map_docling_table(self, table_obj: object) -> Table:
		"""Map a Docling table object into the project Table schema."""
		headers: list[str] = []
		rows: list[list[str]] = []

		export_to_dataframe = getattr(table_obj, "export_to_dataframe", None)
		if callable(export_to_dataframe):
			dataframe = export_to_dataframe()
			headers = [self._safe_cell(col) for col in list(dataframe.columns)]
			rows = [
				[self._safe_cell(cell) for cell in row]
				for row in dataframe.fillna("").values.tolist()
			]
			return Table(headers=headers, rows=rows, title=None)

		return Table(headers=headers, rows=rows, title=None)

	def _safe_cell(self, cell: object) -> str:
		if cell is None:
			return ""
		return str(cell).strip()
