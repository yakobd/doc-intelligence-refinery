import hashlib
import importlib
import logging
import os
from pathlib import Path
from typing import Any

import pdfplumber

from src.models.document_schema import (
    BBox,
    DocumentProfile,
    LDU,
    NormalizedOutput,
    PageIndexNode,
    ProvenanceChain,
)
from src.strategies.base_strategy import BaseStrategy


logger = logging.getLogger(__name__)


class StrategyB(BaseStrategy):
    """Layout-aware extraction strategy with a Docling adapter and robust fallback."""
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)

    def extract(self, pdf_path: str, profile: DocumentProfile) -> NormalizedOutput:
        # FORCE stable pdfplumber fallback (memory-safe for Week 3 submission)
        fallback = self._fallback_extract_with_pdfplumber(pdf_path=pdf_path, profile=profile)
        fallback.metadata["warning"] = "Docling disabled for stability; pdfplumber fallback used"
        return fallback

        converter_cls = self._load_docling_converter()
        if converter_cls is not None:
            try:
                return self._extract_with_docling(pdf_path=pdf_path, profile=profile, converter_cls=converter_cls)
            except Exception as docling_error:
                logger.exception("StrategyB Docling path failed. Falling back to pdfplumber: %s", docling_error)
                fallback = self._fallback_extract_with_pdfplumber(pdf_path=pdf_path, profile=profile)
                fallback.metadata["warning"] = f"Docling adapter failed; fallback used: {docling_error}"
                return fallback

        fallback = self._fallback_extract_with_pdfplumber(pdf_path=pdf_path, profile=profile)
        fallback.metadata["warning"] = "Docling not available; pdfplumber fallback used"
        return fallback

    def _extract_with_docling(self, pdf_path: str, profile: DocumentProfile, converter_cls: Any) -> NormalizedOutput:
        filename = Path(pdf_path).name
        doc_id = Path(pdf_path).stem

        ldus: list[LDU] = []
        provenance_items: list[dict[str, Any]] = []
        warnings: list[str] = []

        converter = converter_cls()

        # MEMORY FIX - reduce batch size to prevent bad_alloc on Windows laptops
        try:
            if hasattr(converter, "pipeline_options"):
                converter.pipeline_options.page_batch_size = 4
            if hasattr(converter, "ocr_options"):
                converter.ocr_options.batch_size = 4
        except:
            pass  # safe fallback

        result = converter.convert(pdf_path)


        document = result.document

        ordered_elements = self._collect_and_sort_docling_elements(document=document)

        for element in ordered_elements:
            try:
                if not element["content"].strip():
                    continue

                element_hash = self._hash_content(element["content"])
                bbox = element["bbox"]
                uid = f"{doc_id}-p{bbox.page_number:03d}-{element['kind']}-{element['ordinal']:03d}"

                ldus.append(
                    LDU(
                        uid=uid,
                        unit_type=element["kind"],
                        content=element["content"],
                        content_hash=element_hash,
                        page_refs=[bbox.page_number],
                        bounding_box=bbox,
                        parent_section=f"page-{bbox.page_number}",
                        child_chunks=self.chunk_text(element["content"]),
                    )
                )
                provenance_items.append(
                    ProvenanceChain(
                        source_file=filename,
                        content_hash=element_hash,
                        bbox=bbox,
                        strategy_used=profile.selected_strategy.value,
                    ).model_dump()
                )
            except Exception as item_error:
                message = f"StrategyB skipped element {element.get('ordinal')} due to error: {item_error}"
                logger.exception(message)
                warnings.append(message)

        index_nodes = self._index_nodes_from_ldus(ldus)

        metadata: dict[str, Any] = {
            "selected_strategy": profile.selected_strategy.value,
            "avg_confidence": 0.9,
            "provenance_chain": provenance_items,
        }
        if warnings:
            metadata["warning"] = " | ".join(warnings)

        return NormalizedOutput(
            filename=filename,
            doc_id=doc_id,
            profile=profile,
            ldus=ldus,
            index=index_nodes,
            metadata=metadata,
        )

    def _collect_and_sort_docling_elements(self, document: Any) -> list[dict[str, Any]]:
        elements: list[dict[str, Any]] = []
        ordinal = 0

        header_elements = self._collect_docling_heading_elements(document=document)
        for header in header_elements:
            ordinal += 1
            header["ordinal"] = ordinal
            elements.append(header)

        markdown = document.export_to_markdown() if hasattr(document, "export_to_markdown") else ""
        if markdown.strip():
            ordinal += 1
            elements.append(
                {
                    "kind": "paragraph",
                    "content": markdown,
                    "bbox": BBox(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0, page_number=1),
                    "top": 0.0,
                    "left": 0.0,
                    "ordinal": ordinal,
                }
            )

        for table_obj in getattr(document, "tables", []) or []:
            table_text = self._table_to_text(table_obj)
            if not table_text:
                continue

            ordinal += 1
            bbox = self._bbox_from_docling_item(table_obj)
            elements.append(
                {
                    "kind": "table",
                    "content": table_text,
                    "bbox": bbox,
                    "top": bbox.y_min,
                    "left": bbox.x_min,
                    "ordinal": ordinal,
                }
            )

        for figure_obj in getattr(document, "figures", []) or []:
            figure_text = self._figure_to_text(figure_obj)
            if not figure_text:
                continue

            ordinal += 1
            bbox = self._bbox_from_docling_item(figure_obj)
            elements.append(
                {
                    "kind": "figure",
                    "content": figure_text,
                    "bbox": bbox,
                    "top": bbox.y_min,
                    "left": bbox.x_min,
                    "ordinal": ordinal,
                }
            )

        elements.sort(key=lambda item: (item["bbox"].page_number, item["top"], item["left"], item["ordinal"]))
        return elements

    def _collect_docling_heading_elements(self, document: Any) -> list[dict[str, Any]]:
        """Extract explicit heading/title nodes from Docling output for hierarchical indexing."""
        elements: list[dict[str, Any]] = []

        for container_name in ("texts", "paragraphs", "items", "elements", "blocks"):
            container = getattr(document, container_name, None)
            if not container:
                continue

            for item in container:
                kind_hint = self._docling_kind_hint(item)
                if kind_hint not in {"section_header", "title", "heading", "header"}:
                    continue

                content = self._docling_item_text(item)
                if not content:
                    continue

                bbox = self._bbox_from_docling_item(item)
                mapped_kind = "title" if kind_hint == "title" else "header"
                elements.append(
                    {
                        "kind": mapped_kind,
                        "content": content,
                        "bbox": bbox,
                        "top": bbox.y_min,
                        "left": bbox.x_min,
                        "ordinal": 0,
                    }
                )

        return elements

    def _docling_kind_hint(self, item: Any) -> str:
        candidates: list[str] = []
        for attr in ("kind", "type", "label", "category", "role"):
            value = getattr(item, attr, None)
            if value is not None:
                candidates.append(str(value))

        if isinstance(item, dict):
            for key in ("kind", "type", "label", "category", "role"):
                value = item.get(key)
                if value is not None:
                    candidates.append(str(value))

        normalized = " ".join(candidates).casefold()
        if "section_header" in normalized:
            return "section_header"
        if "title" in normalized:
            return "title"
        if "heading" in normalized:
            return "heading"
        if "header" in normalized:
            return "header"
        return ""

    def _docling_item_text(self, item: Any) -> str:
        for attr in ("text", "content", "value", "caption", "label"):
            value = getattr(item, attr, None)
            if value and str(value).strip():
                return str(value).strip()

        export_to_text = getattr(item, "export_to_text", None)
        if callable(export_to_text):
            value = export_to_text()
            if value and str(value).strip():
                return str(value).strip()

        if isinstance(item, dict):
            for key in ("text", "content", "value", "caption", "label"):
                value = item.get(key)
                if value and str(value).strip():
                    return str(value).strip()

        return ""

    def _bbox_from_docling_item(self, item: Any) -> BBox:
        for attr_name in ("bbox", "prov", "provenance"):
            candidate = getattr(item, attr_name, None)
            if candidate is None:
                continue

            if hasattr(candidate, "bbox"):
                candidate = candidate.bbox

            values = self._coerce_bbox_values(candidate)
            if values is not None:
                x_min, y_min, x_max, y_max, page_number = values
                return BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, page_number=page_number)

        return BBox(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0, page_number=1)

    def _coerce_bbox_values(self, candidate: Any) -> tuple[float, float, float, float, int] | None:
        if isinstance(candidate, (list, tuple)) and len(candidate) >= 4:
            return float(candidate[0]), float(candidate[1]), float(candidate[2]), float(candidate[3]), 1

        if hasattr(candidate, "x0") and hasattr(candidate, "y0") and hasattr(candidate, "x1") and hasattr(candidate, "y1"):
            page_number = int(getattr(candidate, "page_no", getattr(candidate, "page_number", 1)) or 1)
            return float(candidate.x0), float(candidate.y0), float(candidate.x1), float(candidate.y1), page_number

        if isinstance(candidate, dict):
            keys = ("x0", "y0", "x1", "y1")
            if all(key in candidate for key in keys):
                page_number = int(candidate.get("page_no", candidate.get("page_number", 1)) or 1)
                return (
                    float(candidate["x0"]),
                    float(candidate["y0"]),
                    float(candidate["x1"]),
                    float(candidate["y1"]),
                    page_number,
                )
        return None

    def _figure_to_text(self, figure_obj: Any) -> str:
        for attr in ("caption", "text", "label"):
            value = getattr(figure_obj, attr, None)
            if value and str(value).strip():
                return str(value).strip()

        if isinstance(figure_obj, dict):
            for key in ("caption", "text", "label"):
                value = figure_obj.get(key)
                if value and str(value).strip():
                    return str(value).strip()

        return "figure"

    def _load_docling_converter(self) -> Any:
        try:
            module = importlib.import_module("docling.document_converter")
            return getattr(module, "DocumentConverter", None)
        except Exception:
            return None

    def _fallback_extract_with_pdfplumber(self, pdf_path: str, profile: DocumentProfile) -> NormalizedOutput:
        filename = Path(pdf_path).name
        doc_id = Path(pdf_path).stem

        ldus: list[LDU] = []
        provenance_items: list[dict[str, Any]] = []
        warnings: list[str] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    try:
                        page_text = (page.extract_text() or "").strip()
                        bbox = self._tuple_to_bbox(page.bbox, page_number)

                        if page_text:
                            text_hash = self._hash_content(page_text)
                            ldus.append(
                                LDU(
                                    uid=f"{doc_id}-p{page_number:03d}-text",
                                    unit_type="paragraph",
                                    content=page_text,
                                    content_hash=text_hash,
                                    page_refs=[page_number],
                                    bounding_box=bbox,
                                    parent_section=f"page-{page_number}",
                                    child_chunks=self.chunk_text(page_text),
                                )
                            )
                            provenance_items.append(
                                ProvenanceChain(
                                    source_file=filename,
                                    content_hash=text_hash,
                                    bbox=bbox,
                                    strategy_used=profile.selected_strategy.value,
                                ).model_dump()
                            )

                        for table_index, table in enumerate(page.find_tables(), start=1):
                            rows = table.extract() or []
                            table_text = "\n".join([" | ".join([str(cell or "").strip() for cell in row]) for row in rows]).strip()
                            if not table_text:
                                continue

                            table_bbox = self._tuple_to_bbox(getattr(table, "bbox", page.bbox), page_number)
                            table_hash = self._hash_content(table_text)
                            ldus.append(
                                LDU(
                                    uid=f"{doc_id}-p{page_number:03d}-tbl{table_index:02d}",
                                    unit_type="table",
                                    content=table_text,
                                    content_hash=table_hash,
                                    page_refs=[page_number],
                                    bounding_box=table_bbox,
                                    parent_section=f"page-{page_number}",
                                    child_chunks=self.chunk_text(table_text),
                                )
                            )
                            provenance_items.append(
                                ProvenanceChain(
                                    source_file=filename,
                                    content_hash=table_hash,
                                    bbox=table_bbox,
                                    strategy_used=profile.selected_strategy.value,
                                ).model_dump()
                            )
                    except BaseException as page_error:
                        message = f"StrategyB skipped page {page_number} due to error: {page_error}"
                        logger.exception(message)
                        warnings.append(message)
        except Exception as extraction_error:
            message = f"StrategyB extraction failed with partial output: {extraction_error}"
            logger.exception(message)
            warnings.append(message)

        index_nodes = self._index_nodes_from_ldus(ldus)
        metadata: dict[str, Any] = {
            "selected_strategy": profile.selected_strategy.value,
            "avg_confidence": 0.85,
            "provenance_chain": provenance_items,
        }
        if warnings:
            metadata["warning"] = " | ".join(warnings)

        return NormalizedOutput(
            filename=filename,
            doc_id=doc_id,
            profile=profile,
            ldus=ldus,
            index=index_nodes,
            metadata=metadata,
        )

    def _index_nodes_from_ldus(self, ldus: list[LDU]) -> list[PageIndexNode]:
        page_refs = sorted({page for ldu in ldus for page in ldu.page_refs})
        if not page_refs:
            return [PageIndexNode(title="Page 1", page_start=1, page_end=1, children=[])]

        return [
            PageIndexNode(title=f"Page {page_number}", page_start=page_number, page_end=page_number, children=[])
            for page_number in page_refs
        ]

    def _tuple_to_bbox(self, bbox_tuple: Any, page_number: int) -> BBox:
        if not bbox_tuple or len(bbox_tuple) != 4:
            return BBox(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0, page_number=page_number)

        x_min, y_min, x_max, y_max = bbox_tuple
        return BBox(
            x_min=float(x_min),
            y_min=float(y_min),
            x_max=float(x_max),
            y_max=float(y_max),
            page_number=page_number,
        )

    def _table_to_text(self, table_obj: object) -> str:
        export_to_dataframe = getattr(table_obj, "export_to_dataframe", None)
        if not callable(export_to_dataframe):
            return ""

        dataframe = export_to_dataframe().fillna("")
        headers = [str(col).strip() for col in list(dataframe.columns)]
        rows = dataframe.values.tolist()
        serialized_rows = [" | ".join([str(cell).strip() for cell in row]) for row in rows]
        return "\n".join([" | ".join(headers)] + serialized_rows).strip()

    def _hash_content(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()
