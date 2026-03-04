import base64
import hashlib
import importlib
import io
import json
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
from src.utils.config_loader import get_extraction_config

logger = logging.getLogger(__name__)


class StrategyC(BaseStrategy):
    """Vision-assisted extraction strategy for scanned or difficult PDFs."""

    COST_PER_PAGE = 0.01
    DEFAULT_MODEL = "google/gemini-flash-1.5"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        extraction_config = get_extraction_config(self.config)
        strategy_config = extraction_config.get("strategy_c", {}) if isinstance(extraction_config, dict) else {}

        self.max_budget = float(
            strategy_config.get(
                "vlm_max_spend_per_doc",
                strategy_config.get("max_budget_per_doc", 0.5),
            )
        )
        self.max_pages_to_process = int(strategy_config.get("max_pages_to_process", 3))
        self.default_model = str(strategy_config.get("model", self.DEFAULT_MODEL))
        self.api_key_env = str(strategy_config.get("api_key_env", "OPENROUTER_API_KEY"))

    def extract(self, pdf_path: str, profile: DocumentProfile) -> NormalizedOutput:
        filename = Path(pdf_path).name
        doc_id = Path(pdf_path).stem
        page_payloads: list[dict[str, Any]] = []
        warnings: list[str] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                page_payloads = self._build_page_payloads(pdf_path=pdf_path, pdf=pdf, warnings=warnings)
        except Exception as open_error:
            logger.exception("StrategyC failed to open PDF, returning partial output: %s", open_error)
            return self._partial_output(filename=filename, doc_id=doc_id, profile=profile, warnings=[str(open_error)])

        projected_cost = total_pages * self.COST_PER_PAGE
        if projected_cost > self.max_budget:
            raise ValueError(
                f"Budget exceeded for {filename}: projected ${projected_cost:.2f} > max ${self.max_budget:.2f}"
            )

        try:
            response_payload = self._call_openrouter(page_payloads=page_payloads)
            ldus, provenance_items, index_nodes = self._map_response_to_ldus(response_payload, doc_id, filename, profile)
        except Exception as api_error:
            logger.exception("StrategyC API call failed, returning partial output: %s", api_error)
            warnings.append(str(api_error))
            return self._partial_output(filename=filename, doc_id=doc_id, profile=profile, warnings=warnings)

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

    def _build_page_payloads(self, pdf_path: str, pdf: pdfplumber.PDF, warnings: list[str]) -> list[dict[str, Any]]:
        page_payloads: list[dict[str, Any]] = []
        pages_to_process = min(self.max_pages_to_process, len(pdf.pages))

        for page_number in range(1, pages_to_process + 1):
            try:
                page = pdf.pages[page_number - 1]
                text = page.extract_text() or ""
                image_count = len(page.images or [])
                high_confidence_text = len(text) > 500 and image_count == 0

                payload: dict[str, Any] = {
                    "page_number": page_number,
                    "text": text,
                }

                if not high_confidence_text:
                    image_b64 = self._page_image_base64(pdf_path=pdf_path, page_number=page_number)
                    if image_b64:
                        payload["image_base64"] = image_b64

                page_payloads.append(payload)
            except Exception as page_error:
                message = f"StrategyC skipped page {page_number} due to error: {page_error}"
                logger.exception(message)
                warnings.append(message)

        return page_payloads

    def _page_image_base64(self, pdf_path: str, page_number: int) -> str | None:
        convert_from_path = self._load_pdf2image_converter()
        if convert_from_path is None:
            return None

        try:
            images = convert_from_path(
                pdf_path,
                first_page=page_number,
                last_page=page_number,
                fmt="png",
            )
            if not images:
                return None

            buffer = io.BytesIO()
            images[0].save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception:
            return None

    def _call_openrouter(self, page_payloads: list[dict[str, Any]]) -> dict[str, Any]:
        httpx_module = self._load_httpx()
        if httpx_module is None:
            raise ValueError("httpx is required for StrategyC.")

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            return {"pages": []}

        model_name = os.getenv("OPENROUTER_MODEL", self.default_model)
        instruction = (
            "Extract the main text and any tables into a structured JSON format. "
            "Return strict JSON with this schema: "
            "{'pages':[{'page_number':int,'text':str,'tables':[{'title':str|null,'headers':[str],'rows':[[str]]}]}]}"
        )

        content_items: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        for payload in page_payloads:
            content_items.append({"type": "text", "text": f"Page {payload['page_number']} text:\n{payload.get('text', '')}"})
            image_base64 = payload.get("image_base64")
            if image_base64:
                content_items.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})

        request_body = {
            "model": model_name,
            "messages": [{"role": "user", "content": content_items}],
            "temperature": 0,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "doc-intelligence-refinery",
        }

        with httpx_module.Client(timeout=120.0) as client:
            response = client.post(self.OPENROUTER_URL, headers=headers, json=request_body)
            response.raise_for_status()
            payload = response.json()

        message_content = self._extract_message_content(payload)
        return self._coerce_json_payload(message_content)

    def _extract_message_content(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    def _coerce_json_payload(self, raw_text: str) -> dict[str, Any]:
        try:
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception:
            return {"pages": []}

    def _load_httpx(self) -> Any:
        try:
            return importlib.import_module("httpx")
        except Exception:
            return None

    def _load_pdf2image_converter(self) -> Any:
        try:
            module = importlib.import_module("pdf2image")
            return getattr(module, "convert_from_path", None)
        except Exception:
            return None

    def _map_response_to_ldus(
        self,
        response_payload: dict[str, Any],
        doc_id: str,
        filename: str,
        profile: DocumentProfile,
    ) -> tuple[list[LDU], list[dict[str, Any]], list[PageIndexNode]]:
        pages_data = response_payload.get("pages", [])
        ldus: list[LDU] = []
        provenance_items: list[dict[str, Any]] = []
        index_nodes: list[PageIndexNode] = []

        if not isinstance(pages_data, list):
            pages_data = []

        for page_item in pages_data:
            if not isinstance(page_item, dict):
                continue

            page_number = int(page_item.get("page_number", len(index_nodes) + 1))
            page_text = str(page_item.get("text", "")).strip()
            page_bbox = BBox(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0, page_number=page_number)

            if page_text:
                text_hash = self._hash_content(page_text)
                ldu_uid = f"{doc_id}-p{page_number:03d}-text"
                ldus.append(
                    LDU(
                        uid=ldu_uid,
                        unit_type="paragraph",
                        content=page_text,
                        content_hash=text_hash,
                        page_refs=[page_number],
                        bounding_box=page_bbox,
                        parent_section=f"page-{page_number}",
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

            for table_idx, table_item in enumerate(page_item.get("tables", []), start=1):
                if not isinstance(table_item, dict):
                    continue

                headers = table_item.get("headers", [])
                rows = table_item.get("rows", [])
                table_text = self._serialize_table(headers, rows)
                if not table_text:
                    continue

                table_hash = self._hash_content(table_text)
                table_uid = f"{doc_id}-p{page_number:03d}-tbl{table_idx:02d}"
                ldus.append(
                    LDU(
                        uid=table_uid,
                        unit_type="table",
                        content=table_text,
                        content_hash=table_hash,
                        page_refs=[page_number],
                        bounding_box=page_bbox,
                        parent_section=f"page-{page_number}",
                        child_chunks=self.chunk_text(table_text),
                    )
                )
                provenance_items.append(
                    ProvenanceChain(
                        source_file=filename,
                        content_hash=table_hash,
                        bbox=page_bbox,
                        strategy_used=profile.selected_strategy.value,
                    ).model_dump()
                )

            index_nodes.append(
                PageIndexNode(
                    title=f"Page {page_number}",
                    page_start=page_number,
                    page_end=page_number,
                    children=[],
                )
            )

        if not index_nodes:
            index_nodes.append(PageIndexNode(title="Page 1", page_start=1, page_end=1, children=[]))

        return ldus, provenance_items, index_nodes

    def _serialize_table(self, headers: Any, rows: Any) -> str:
        header_text = " | ".join([str(cell).strip() for cell in headers]) if isinstance(headers, list) else ""

        row_texts: list[str] = []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, list):
                    row_texts.append(" | ".join([str(cell).strip() for cell in row]))

        text = "\n".join([line for line in [header_text, *row_texts] if line.strip()])
        return text.strip()

    def _hash_content(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _partial_output(self, filename: str, doc_id: str, profile: DocumentProfile, warnings: list[str]) -> NormalizedOutput:
        metadata: dict[str, Any] = {
            "selected_strategy": profile.selected_strategy.value,
            "avg_confidence": 0.0,
            "provenance_chain": [],
            "warning": " | ".join(warnings) if warnings else "StrategyC returned partial output",
        }
        return NormalizedOutput(
            filename=filename,
            doc_id=doc_id,
            profile=profile,
            ldus=[],
            index=[PageIndexNode(title="Page 1", page_start=1, page_end=1, children=[])],
            metadata=metadata,
        )
