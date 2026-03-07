from pathlib import Path
from typing import Any
import logging
import os

import pdfplumber

from src.agents.domain_classifier import BaseDomainClassifier, KeywordDomainClassifier
from src.models.document_schema import (
    DocumentProfile,
    LayoutComplexity,
    OriginType,
    StrategyTier,
)
from src.utils.config_loader import get_triage_config, load_config


logger = logging.getLogger(__name__)


class TriageAgent:
    """Profiles documents and routes them toward an extraction strategy."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str = "rubric/extraction_rules.yaml",
        domain_classifier: BaseDomainClassifier | None = None,
    ) -> None:
        self.config = config or load_config(config_path)
        self.rules = self._load_rules()
        self.thresholds = self.rules["thresholds"]
        self.domain_keywords = self.rules["domain_keywords"]
        self.cost_tiers = self.rules["cost_tiers"]
        self.mixed_mode_ratio = float(
            self.thresholds.get(
                "mixed_mode_ratio",
                self.thresholds.get("mixed_origin_low_density_page_ratio", 0.25),
            )
        )
        self.high_font_density_char_count = int(self.thresholds.get("high_font_density_char_count", 200))
        self.scanned_char_density_per_area = float(self.thresholds.get("scanned_char_density_per_area", 0.0002))
        self.table_to_text_ratio_threshold = float(self.thresholds.get("table_to_text_ratio_threshold", 0.2))
        self.scan_chars_per_page_threshold = float(self.thresholds.get("scan_chars_per_page_threshold", 150))
        self.domain_classifier = domain_classifier or KeywordDomainClassifier(self.domain_keywords)


    def profile_document(self, pdf_path: str) -> DocumentProfile:
        filename = Path(pdf_path).name

        with pdfplumber.open(pdf_path) as pdf:
            signals = self._collect_pdf_signals(pdf)

        origin_type = self._detect_origin(signals)
        layout_complexity = self._detect_layout_complexity(signals)
        domain_hint, domain_confidence = self._classify_domain(signals["combined_text"])
        estimated_chars = int(signals.get("estimated_chars", 0))
        selected_strategy = self._select_strategy(
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            signals=signals,
            estimated_chars=estimated_chars,
            pages=signals["total_pages"],
        )
        confidence_score = self._calculate_confidence(signals, origin_type)
        estimated_cost = self._estimate_cost(
            selected_strategy=selected_strategy,
            estimated_chars=estimated_chars,
            pages=signals["total_pages"],
        )

        if selected_strategy == StrategyTier.STRATEGY_B:
            # Ensure layout strategy can use Docling path without manual env toggling.
            os.environ["ENABLE_DOCLING_ADAPTER"] = "1"

        return DocumentProfile(
            filename=filename,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            selected_strategy=selected_strategy,
            confidence_score=confidence_score,
            estimated_cost=estimated_cost,
            estimated_chars=estimated_chars,
            pages=signals["total_pages"],
            language="en",
            domain_hint=domain_hint,
            is_form_fillable=signals["has_acroform"],
            domain_confidence=domain_confidence,
        )

    def _load_rules(self) -> dict[str, Any]:
        triage_config = get_triage_config(self.config)
        if not isinstance(triage_config, dict):
            raise ValueError("Missing triage_config in rubric/extraction_rules.yaml")

        required_sections = ["thresholds", "domain_keywords", "cost_tiers"]
        for section in required_sections:
            if section not in triage_config:
                raise ValueError(f"Missing '{section}' in rubric/extraction_rules.yaml triage_config")

        return triage_config

    def _collect_pdf_signals(self, pdf: pdfplumber.PDF) -> dict[str, Any]:
        total_pages = len(pdf.pages)
        if total_pages == 0:
            return {
                "total_pages": 1,
                "has_fonts_any": False,
                "low_density_ratio": 1.0,
                "avg_chars_per_page": 0.0,
                "estimated_chars": 0,
                "avg_char_density": 0.0,
                "avg_bbox_density": 0.0,
                "table_page_ratio": 0.0,
                "table_to_text_ratio": 0.0,
                "gutter_found": False,
                "combined_text": "",
                "has_acroform": self._has_acroform(pdf),
                "text_rich_pages": 0,
                "zero_font_pages": [1],
                "high_font_density_pages": [],
            }

        combined_text_parts: list[str] = []
        fonts_detected_pages = 0
        low_density_pages = 0
        table_pages = 0
        text_rich_pages = 0

        total_chars = 0
        total_char_density = 0.0
        total_bbox_density = 0.0
        total_table_chars = 0
        gutter_found = False
        zero_font_pages: list[int] = []
        high_font_density_pages: list[int] = []

        for page_index, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
                combined_text_parts.append(text)

                char_count = len(text)
                total_chars += char_count

                chars = page.objects.get("char", [])
                font_chars_count = sum(1 for char in chars if char.get("fontname"))
                if font_chars_count > 0:
                    fonts_detected_pages += 1
                else:
                    zero_font_pages.append(page_index)

                if font_chars_count >= self.high_font_density_char_count:
                    high_font_density_pages.append(page_index)

                if char_count <= float(self.thresholds["scanned_image_char_density"]):
                    low_density_pages += 1

                detected_tables = page.find_tables()
                if bool(detected_tables):
                    table_pages += 1

                for detected_table in detected_tables:
                    raw_rows = detected_table.extract() or []
                    for row in raw_rows:
                        for cell in row:
                            if cell:
                                total_table_chars += len(str(cell).strip())

                if char_count >= float(self.thresholds["single_column_min_chars_per_page"]):
                    text_rich_pages += 1

                page_area = max(1.0, float(page.width) * float(page.height))
                total_char_density += char_count / page_area
                char_bbox_area = 0.0
                for char in chars:
                    x0 = float(char.get("x0", 0.0) or 0.0)
                    x1 = float(char.get("x1", 0.0) or 0.0)
                    top = float(char.get("top", 0.0) or 0.0)
                    bottom = float(char.get("bottom", top) or top)

                    width = max(0.0, x1 - x0)
                    height = max(0.0, bottom - top)
                    char_bbox_area += width * height

                bbox_density = char_bbox_area / page_area
                total_bbox_density += bbox_density

                if not gutter_found and self._has_vertical_gutter(chars=chars, page_height=float(page.height)):
                    gutter_found = True
            except BaseException as page_error:
                logger.warning("Triage skipped a problematic page while collecting signals: %s", page_error)
                low_density_pages += 1
                continue

        return {
            "total_pages": total_pages,
            "has_fonts_any": fonts_detected_pages > 0,
            "low_density_ratio": low_density_pages / total_pages,
            "avg_chars_per_page": total_chars / total_pages,
            "estimated_chars": total_chars,
            "avg_char_density": total_char_density / total_pages,
            "avg_bbox_density": total_bbox_density / total_pages,
            "table_page_ratio": table_pages / total_pages,
            "table_pages": table_pages,
            "table_to_text_ratio": (total_table_chars / max(1, total_chars)),
            "gutter_found": gutter_found,
            "combined_text": "\n".join(combined_text_parts),
            "has_acroform": self._has_acroform(pdf),
            "text_rich_pages": text_rich_pages,
            "zero_font_pages": zero_font_pages,
            "high_font_density_pages": high_font_density_pages,
        }

    def _has_acroform(self, pdf: pdfplumber.PDF) -> bool:
        doc = getattr(pdf, "doc", None)
        catalog = getattr(doc, "catalog", {}) or {}
        trailer = getattr(doc, "trailer", {}) or {}

        if "AcroForm" in catalog or "/AcroForm" in catalog:
            return True

        root = trailer.get("Root") if isinstance(trailer, dict) else None
        if root is None:
            root = trailer.get("/Root") if isinstance(trailer, dict) else None

        if isinstance(root, dict):
            return ("AcroForm" in root) or ("/AcroForm" in root)

        for attr in ("get",):
            getter = getattr(root, attr, None)
            if callable(getter):
                if getter("AcroForm") is not None or getter("/AcroForm") is not None:
                    return True

        return False

    def _detect_origin(self, signals: dict[str, Any]) -> OriginType:
        zero_font_pages = signals.get("zero_font_pages", [])
        high_font_density_pages = signals.get("high_font_density_pages", [])
        avg_char_density = float(signals.get("avg_char_density", 0.0))
        scan_like_density = avg_char_density <= self.scanned_char_density_per_area

        if zero_font_pages and high_font_density_pages:
            logger.info(
                "Mixed-mode document detected with zero-font pages=%s and high-font pages=%s",
                self._compress_page_ranges(zero_font_pages),
                self._compress_page_ranges(high_font_density_pages),
            )
            return OriginType.MIXED

        if scan_like_density:
            if signals["has_fonts_any"] and signals["low_density_ratio"] >= self.mixed_mode_ratio:
                return OriginType.MIXED
            if signals["has_acroform"]:
                return OriginType.MIXED
            return OriginType.SCANNED_IMAGE

        if not signals["has_fonts_any"]:
            return OriginType.SCANNED_IMAGE

        if signals["low_density_ratio"] >= self.mixed_mode_ratio:
            return OriginType.MIXED

        return OriginType.NATIVE_DIGITAL

    def _detect_layout_complexity(self, signals: dict[str, Any]) -> LayoutComplexity:
        if (
            signals["table_page_ratio"] >= float(self.thresholds["table_heavy_density"])
            or float(signals.get("table_to_text_ratio", 0.0)) >= self.table_to_text_ratio_threshold
        ):
            return LayoutComplexity.TABLE_HEAVY

        if (
            signals["gutter_found"]
            and signals["avg_chars_per_page"] >= float(self.thresholds["multi_column_min_chars_per_page"])
        ):
            return LayoutComplexity.MULTI_COLUMN

        if (
            signals["avg_chars_per_page"] >= float(self.thresholds["single_column_min_chars_per_page"])
            and signals["avg_bbox_density"] >= float(self.thresholds["single_column_min_bbox_density"])
        ):
            return LayoutComplexity.SINGLE_COLUMN

        return LayoutComplexity.MULTI_COLUMN

    def _has_vertical_gutter(self, chars: list, page_height: float) -> bool:
        if not chars or page_height <= 0:
            return False

        gutter_x_min = float(self.thresholds["multi_column_gutter_x_min"])
        gutter_x_max = float(self.thresholds["multi_column_gutter_x_max"])

        left_char_count = 0
        right_char_count = 0
        gutter_vertical_spans: list[tuple[float, float]] = []

        for char in chars:
            x0 = float(char.get("x0", 0.0) or 0.0)
            x1 = float(char.get("x1", 0.0) or 0.0)
            top = float(char.get("top", 0.0) or 0.0)
            bottom = float(char.get("bottom", top) or top)

            if x1 < gutter_x_min:
                left_char_count += 1
            if x0 > gutter_x_max:
                right_char_count += 1

            intersects_gutter = not (x1 <= gutter_x_min or x0 >= gutter_x_max)
            if intersects_gutter:
                span_start = max(0.0, min(top, page_height))
                span_end = max(0.0, min(bottom, page_height))
                if span_end > span_start:
                    gutter_vertical_spans.append((span_start, span_end))

        if left_char_count < int(self.thresholds["multi_column_min_left_chars"]):
            return False
        if right_char_count < int(self.thresholds["multi_column_min_right_chars"]):
            return False

        slice_count = int(self.thresholds["multi_column_slice_count"])
        slice_height = page_height / slice_count
        empty_slices = 0

        for idx in range(slice_count):
            slice_start = idx * slice_height
            slice_end = slice_start + slice_height
            has_gutter_char = any(
                not (span_end <= slice_start or span_start >= slice_end)
                for span_start, span_end in gutter_vertical_spans
            )
            if not has_gutter_char:
                empty_slices += 1

        empty_ratio = empty_slices / slice_count
        return empty_ratio >= float(self.thresholds["multi_column_empty_slice_ratio"])

    def _classify_domain(self, text: str) -> tuple[str, float]:
        """Pluggable domain classifier entry point.

        Replace this single method to swap keyword-based classification with VLM/ML.
        """
        domain_hint, domain_confidence = self.domain_classifier.classify(text)
        domain = str(domain_hint).strip().lower() or "general"
        confidence = float(domain_confidence) if isinstance(domain_confidence, (int, float)) else 0.0
        return domain, max(0.0, min(1.0, round(confidence, 4)))

    def _select_strategy(
        self,
        origin_type: OriginType,
        layout_complexity: LayoutComplexity,
        signals: dict[str, Any],
        estimated_chars: int,
        pages: int,
    ) -> StrategyTier:
        # Threshold detection for likely scanned documents (from YAML-configurable heuristic).
        safe_pages = max(1, int(pages))
        chars_per_page = float(max(0, int(estimated_chars))) / float(safe_pages)
        if chars_per_page < self.scan_chars_per_page_threshold:
            return StrategyTier.STRATEGY_C

        if origin_type in {OriginType.SCANNED_IMAGE, OriginType.MIXED}:
            return StrategyTier.STRATEGY_C

        table_pages = int(signals.get("table_pages", 0) or 0)
        if layout_complexity in {LayoutComplexity.TABLE_HEAVY, LayoutComplexity.MULTI_COLUMN} or table_pages >= 2:
            return StrategyTier.STRATEGY_B

        return StrategyTier.STRATEGY_A

    def _estimate_cost(self, selected_strategy: StrategyTier, estimated_chars: int, pages: int) -> float:
        strategy_to_key = {
            StrategyTier.STRATEGY_A: "strategy_a",
            StrategyTier.STRATEGY_B: "strategy_b",
            StrategyTier.STRATEGY_C: "strategy_c",
        }
        tier_value = self.cost_tiers[strategy_to_key[selected_strategy]]

        # Backward compatibility: scalar tier values are treated as cost per million chars.
        cost_per_million_chars = float(tier_value) if isinstance(tier_value, (int, float)) else 0.0
        if isinstance(tier_value, dict):
            if isinstance(tier_value.get("cost_per_million_chars"), (int, float)):
                cost_per_million_chars = float(tier_value["cost_per_million_chars"])
            elif isinstance(tier_value.get("cost_per_1k_chars"), (int, float)):
                cost_per_million_chars = float(tier_value["cost_per_1k_chars"]) * 1000.0

        safe_chars = max(0, int(estimated_chars))
        if safe_chars == 0:
            safe_chars = max(1, int(pages)) * int(self.thresholds.get("fallback_chars_per_page", 1200))

        return round((safe_chars / 1_000_000.0) * cost_per_million_chars, 6)

    def _calculate_confidence(self, signals: dict[str, Any], origin_type: OriginType) -> float:
        score = float(self.thresholds["confidence_base"])

        scan_like = signals["low_density_ratio"] >= float(self.thresholds["mixed_origin_low_density_page_ratio"])
        text_rich = signals["text_rich_pages"] > 0

        if signals["has_fonts_any"] and scan_like:
            score -= float(self.thresholds["confidence_penalty_fonts_but_scan_like"])

        if (not signals["has_fonts_any"]) and text_rich:
            score -= float(self.thresholds["confidence_penalty_no_fonts_but_text_rich"])

        if signals["has_acroform"] and origin_type != OriginType.NATIVE_DIGITAL:
            score -= float(self.thresholds["confidence_penalty_form_fillable_scan_like"])

        return max(0.0, min(1.0, round(score, 4)))

    def _compress_page_ranges(self, pages: list[int]) -> str:
        if not pages:
            return "none"

        ordered = sorted(set(pages))
        ranges: list[str] = []
        start = ordered[0]
        end = ordered[0]

        for page in ordered[1:]:
            if page == end + 1:
                end = page
                continue

            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = page
            end = page

        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")

        return ",".join(ranges)
