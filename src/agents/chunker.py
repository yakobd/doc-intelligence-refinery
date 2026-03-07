import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.models.document_schema import BBox, LDU


class ChunkValidator:
    """Validates semantic chunks before they are turned into LDUs."""

    @staticmethod
    def validate(chunk_type: str, content: str, parent_section: Optional[str]) -> Tuple[bool, Optional[str]]:
        normalized_type = (chunk_type or "").strip().lower()
        normalized_content = (content or "").strip()

        if normalized_type == "table":
            # Table chunks must include a header row and at least one data row.
            non_empty_lines = [line for line in normalized_content.splitlines() if line.strip()]
            if len(non_empty_lines) < 2:
                return False, "table chunks must contain at least two lines (header + data)"

        if normalized_type == "figure":
            caption = (parent_section or "").strip()
            if not caption:
                return False, "figure chunks must include a non-empty parent_section caption"

        return True, None


class SemanticChunker:
    """Converts raw extraction segments into validated LDU objects."""

    @staticmethod
    def _normalize_bbox(raw_bbox: Any) -> List[float]:
        if isinstance(raw_bbox, BBox):
            return [
                float(raw_bbox.x_min),
                float(raw_bbox.y_min),
                float(raw_bbox.x_max),
                float(raw_bbox.y_max),
            ]

        if isinstance(raw_bbox, dict):
            if {"x_min", "y_min", "x_max", "y_max"}.issubset(raw_bbox.keys()):
                return [
                    float(raw_bbox["x_min"]),
                    float(raw_bbox["y_min"]),
                    float(raw_bbox["x_max"]),
                    float(raw_bbox["y_max"]),
                ]

        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            return [float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])]

        return [0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def _resolve_page(segment: Dict[str, Any], fallback_page: int = 1) -> int:
        page_value = (
            segment.get("page")
            or segment.get("page_number")
            or segment.get("page_ref")
            or fallback_page
        )
        try:
            return int(page_value)
        except (TypeError, ValueError):
            return fallback_page

    def _generate_content_hash(self, content: str, bbox: List[float], page: int) -> str:
        normalized_content = (content or "").strip()
        normalized_bbox = [round(float(v), 6) for v in (bbox or [0.0, 0.0, 0.0, 0.0])[:4]]
        payload = f"{normalized_content}|{normalized_bbox}|{int(page)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def process_segments(self, raw_extractions: Iterable[Dict[str, Any]]) -> List[LDU]:
        """
        Convert raw extraction segments into validated LDU objects.

        Invalid segments are skipped if they violate semantic chunk validation rules.
        """
        ldus: List[LDU] = []

        for index, segment in enumerate(raw_extractions, start=1):
            if not isinstance(segment, dict):
                continue

            content = str(segment.get("content", "")).strip()
            if not content:
                continue

            chunk_type = str(segment.get("chunk_type") or segment.get("unit_type") or "paragraph").strip()
            parent_section = segment.get("parent_section")
            if parent_section is not None:
                parent_section = str(parent_section)

            is_valid, _ = ChunkValidator.validate(chunk_type=chunk_type, content=content, parent_section=parent_section)
            if not is_valid:
                continue

            raw_bbox = segment.get("bounding_box")
            if raw_bbox is None:
                raw_bbox = segment.get("bbox")
            bbox = self._normalize_bbox(raw_bbox)

            page = self._resolve_page(segment=segment)
            content_hash = self._generate_content_hash(content=content, bbox=bbox, page=page)

            uid = str(segment.get("uid") or f"ldu-p{page:03d}-{index:04d}-{content_hash[:10]}")
            token_count = int(segment.get("token_count", len(content.split())))

            ldu = LDU(
                uid=uid,
                content=content,
                chunk_type=chunk_type,
                content_hash=content_hash,
                page_refs=[page],
                bounding_box=bbox,
                parent_section=parent_section,
                token_count=max(token_count, 0),
                child_chunks=list(segment.get("child_chunks", [])),
            )
            ldus.append(ldu)

        return ldus
