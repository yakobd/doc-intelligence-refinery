import hashlib
import logging
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
from src.utils.config_loader import get_extraction_config

logger = logging.getLogger(__name__)


class StrategyA(BaseStrategy):
    """Fast text-first strategy for native digital and simple-layout PDFs."""

    STANDARD_FONTS = {"times", "arial", "helvetica", "courier", "calibri", "georgia", "verdana"}
    DEFAULT_SCANNED_IMAGE_THRESHOLD = 0.35
    LOW_CONFIDENCE_ON_SCAN = 0.2

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        extraction_config = get_extraction_config(self.config)
        strategy_config = extraction_config.get("strategy_a", {}) if isinstance(extraction_config, dict) else {}
        self.scanned_image_threshold = float(
            strategy_config.get("scanned_image_threshold", self.DEFAULT_SCANNED_IMAGE_THRESHOLD)
        )

    def extract(self, pdf_path: str, profile: DocumentProfile) -> NormalizedOutput:
        filename = Path(pdf_path).name
        doc_id = Path(pdf_path).stem
        ldus: list[LDU] = []
        provenance_items: list[dict[str, Any]] = []
        index_nodes: list[PageIndexNode] = []
        warnings: list[str] = []
        page_confidences: list[float] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    try:
                        page_text = (page.extract_text() or "").strip()
                        chars = page.objects.get("char", [])
                        page_bbox = self._tuple_to_bbox(page.bbox, page_index)

                        confidence_score = self._calculate_weighted_confidence(
                            chars=chars,
                            text=page_text,
                            images=page.images or [],
                            page_width=float(page.width),
                            page_height=float(page.height),
                        )
                        page_confidences.append(confidence_score)

                        if page_text:
                            text_hash = self._hash_content(page_text)
                            ldu_uid = f"{doc_id}-p{page_index:03d}-text"
                            ldus.append(
                                LDU(
                                    uid=ldu_uid,
                                    unit_type="paragraph",
                                    content=page_text,
                                    content_hash=text_hash,
                                    page_refs=[page_index],
                                    bounding_box=page_bbox,
                                    parent_section=f"page-{page_index}",
                                    child_chunks=self.chunk_text(page_text),
                                )
                            )
                            provenance_items.append(
                                ProvenanceChain(
                                    source_file=filename,
                                    content_hash=text_hash,
                                    bbox=page_bbox,
                                    strategy_used=profile.selected_strategy.value,
                                ).model_dump()
                            )

                        for table_idx, detected_table in enumerate(page.find_tables(), start=1):
                            raw_rows = detected_table.extract() or []
                            if not raw_rows:
                                continue

                            table_text = "\n".join([" | ".join([self._safe_cell(cell) for cell in row]) for row in raw_rows])
                            if not table_text.strip():
                                continue

                            table_hash = self._hash_content(table_text)
                            table_bbox = self._tuple_to_bbox(getattr(detected_table, "bbox", page.bbox), page_index)
                            ldu_uid = f"{doc_id}-p{page_index:03d}-tbl{table_idx:02d}"

                            ldus.append(
                                LDU(
                                    uid=ldu_uid,
                                    unit_type="table",
                                    content=table_text,
                                    content_hash=table_hash,
                                    page_refs=[page_index],
                                    bounding_box=table_bbox,
                                    parent_section=f"page-{page_index}",
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

                        index_nodes.append(
                            PageIndexNode(
                                title=f"Page {page_index}",
                                page_start=page_index,
                                page_end=page_index,
                                children=[],
                            )
                        )

                        if confidence_score < 0.5:
                            logger.warning(
                                "Low StrategyA confidence on page %s of %s (confidence=%.2f)",
                                page_index,
                                filename,
                                confidence_score,
                            )
                    except BaseException as page_error:
                        message = f"StrategyA skipped page {page_index} due to error: {page_error}"
                        logger.exception(message)
                        warnings.append(message)
                        continue
        except Exception as extraction_error:
            message = f"StrategyA extraction failed with partial output: {extraction_error}"
            logger.exception(message)
            warnings.append(message)

        if not index_nodes:
            index_nodes = [PageIndexNode(title="Page 1", page_start=1, page_end=1, children=[])]

        metadata: dict[str, Any] = {
            "selected_strategy": profile.selected_strategy.value,
            "avg_confidence": round(sum(page_confidences) / len(page_confidences), 4) if page_confidences else 0.0,
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

    def _calculate_weighted_confidence(
        self,
        chars: list[dict[str, Any]],
        text: str,
        images: list[dict[str, Any]],
        page_width: float,
        page_height: float,
    ) -> float:
        image_density = self._image_density_ratio(images=images, page_width=page_width, page_height=page_height)
        if image_density > self.scanned_image_threshold:
            return self.LOW_CONFIDENCE_ON_SCAN

        font_score = self._font_presence_score(chars)
        density_score = self._character_density_score(text=text, page_width=page_width, page_height=page_height)
        image_ratio_score = self._image_density_score(image_density)

        weighted = (0.35 * font_score) + (0.35 * density_score) + (0.30 * image_ratio_score)
        return max(0.0, min(1.0, round(weighted, 4)))

    def _font_presence_score(self, chars: list[dict[str, Any]]) -> float:
        if not chars:
            return 0.15

        font_names = [str(c.get("fontname", "")).lower() for c in chars if c.get("fontname")]
        if not font_names:
            return 0.25

        standard_hits = sum(1 for name in font_names if any(f in name for f in self.STANDARD_FONTS))
        ratio = standard_hits / len(font_names)
        return 0.5 + (0.5 * ratio)

    def _character_density_score(self, text: str, page_width: float, page_height: float) -> float:
        area = max(1.0, page_width * page_height)
        density = len(text) / area

        if density < 0.00015:
            return 0.2
        if density > 0.012:
            return 0.4
        if 0.00015 <= density <= 0.007:
            return 1.0
        return 0.75

    def _image_density_ratio(self, images: list[dict[str, Any]], page_width: float, page_height: float) -> float:
        total_page_area = max(1.0, page_width * page_height)
        total_image_area = 0.0

        for image in images:
            width = float(image.get("width", 0.0) or 0.0)
            height = float(image.get("height", 0.0) or 0.0)
            total_image_area += max(0.0, width * height)

        return total_image_area / total_page_area

    def _image_density_score(self, image_density: float) -> float:
        if image_density > 0.6:
            return 0.2
        if image_density > 0.3:
            return 0.5
        return 1.0

    def _hash_content(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

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

    def _safe_cell(self, cell: object) -> str:
        if cell is None:
            return ""
        return str(cell).strip()
