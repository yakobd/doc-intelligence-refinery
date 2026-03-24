import base64
import hashlib
import importlib
import io
import json
import logging
import math
import os
import re
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

    DEFAULT_COST_PER_1K_TOKENS = 0.000125
    DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
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
        configured_cost_per_million = strategy_config.get("cost_per_million_tokens")
        if isinstance(configured_cost_per_million, (int, float)):
            self.cost_per_million_tokens = float(configured_cost_per_million)
        else:
            cost_per_1k = strategy_config.get("cost_per_1k_tokens", self.DEFAULT_COST_PER_1K_TOKENS)
            self.cost_per_million_tokens = float(cost_per_1k) * 1000.0

        self.max_pages_to_process = int(strategy_config.get("max_pages_to_process", 3))
        self.default_model = str(strategy_config.get("model", self.DEFAULT_MODEL))
        self.api_key_env = str(strategy_config.get("api_key_env", "OPENROUTER_API_KEY"))

    def extract(self, pdf_path: str, profile: DocumentProfile) -> NormalizedOutput:
        filename = Path(pdf_path).name
        doc_id = Path(pdf_path).stem
        page_payloads: list[dict[str, Any]] = []
        warnings: list[str] = []

        pre_flight_tokens = self._estimate_tokens(
            profile=profile,
            total_pages=int(getattr(profile, "pages", 0) or 0),
        )
        pre_flight_cost = (pre_flight_tokens / 1_000_000.0) * self.cost_per_million_tokens
        if pre_flight_cost > self.max_budget:
            warnings.append(
                f"STRATEGY_C_ABORTED: Pre-flight cost estimate (${pre_flight_cost:.6f}) exceeds budget (${self.max_budget:.6f})"
            )
            return self._partial_output(
                filename=filename,
                doc_id=doc_id,
                profile=profile,
                warnings=warnings,
                projected_cost=pre_flight_cost,
                estimated_tokens=pre_flight_tokens,
            )

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_payloads, estimated_tokens, projected_cost = self._build_page_payloads(
                    pdf_path=pdf_path,
                    pdf=pdf,
                    warnings=warnings,
                )
        except Exception as open_error:
            logger.exception("StrategyC failed to open PDF, returning partial output: %s", open_error)
            return self._partial_output(filename=filename, doc_id=doc_id, profile=profile, warnings=[str(open_error)])

        if not page_payloads:
            if not warnings:
                warnings.append("Budget exceeded, partial results returned.")
            return self._partial_output(
                filename=filename,
                doc_id=doc_id,
                profile=profile,
                warnings=warnings,
                projected_cost=projected_cost,
                estimated_tokens=estimated_tokens,
            )

        try:
            response_payload = self._call_openrouter(page_payloads=page_payloads)
            ldus, provenance_items, index_nodes = self._map_response_to_ldus(
                response_payload=response_payload,
                doc_id=doc_id,
                filename=filename,
                profile=profile,
                page_payloads=page_payloads,
            )
        except Exception as api_error:
            logger.exception("StrategyC API call failed, returning partial output: %s", api_error)
            warnings.append(str(api_error))
            return self._partial_output(
                filename=filename,
                doc_id=doc_id,
                profile=profile,
                warnings=warnings,
                projected_cost=projected_cost,
                estimated_tokens=estimated_tokens,
            )

        metadata: dict[str, Any] = {
            "selected_strategy": profile.selected_strategy.value,
            "avg_confidence": float(self.config.get("router_config", {}).get("default_fallback_confidence", 0.85)),
            "provenance_chain": provenance_items,
            "estimated_tokens": estimated_tokens,
            "projected_cost": round(projected_cost, 6),
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

    def _build_page_payloads(
        self,
        pdf_path: str,
        pdf: pdfplumber.PDF,
        warnings: list[str],
    ) -> tuple[list[dict[str, Any]], int, float]:
        page_payloads: list[dict[str, Any]] = []
        pages_to_process = min(self.max_pages_to_process, len(pdf.pages))
        running_tokens = 0

        for page_number in range(1, pages_to_process + 1):
            try:
                page = pdf.pages[page_number - 1]
                text = page.extract_text() or ""
                image_count = len(page.images or [])
                high_confidence_text = len(text) > 500 and image_count == 0
                include_image = not high_confidence_text

                page_token_estimate = self._estimate_page_tokens(text=text, include_image=include_image)
                candidate_total_tokens = running_tokens + page_token_estimate
                candidate_projected_cost = (candidate_total_tokens / 1_000_000.0) * self.cost_per_million_tokens

                if candidate_projected_cost > self.max_budget:
                    warnings.append("Budget exceeded, partial results returned.")
                    break

                page_width = float(getattr(page, "width", 0.0) or 0.0)
                page_height = float(getattr(page, "height", 0.0) or 0.0)

                payload: dict[str, Any] = {
                    "page_number": page_number,
                    "text": text,
                    "page_width": page_width,
                    "page_height": page_height,
                }

                if include_image:
                    image_b64 = self._page_image_base64(pdf_path=pdf_path, page_number=page_number)
                    if image_b64:
                        payload["image_base64"] = image_b64

                page_payloads.append(payload)
                running_tokens = candidate_total_tokens
            except Exception as page_error:
                message = f"StrategyC skipped page {page_number} due to error: {page_error}"
                logger.exception(message)
                warnings.append(message)

        projected_cost = (running_tokens / 1_000_000.0) * self.cost_per_million_tokens
        return page_payloads, running_tokens, projected_cost

    def _estimate_page_tokens(self, text: str, include_image: bool) -> int:
        text_tokens = int(math.ceil(len(text or "") / 4.0))
        image_tokens = 700 if include_image else 0
        # Small fixed overhead for instruction and schema context.
        return text_tokens + image_tokens + 150

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

    def _estimate_tokens(self, profile: DocumentProfile, total_pages: int) -> int:
        safe_pages = max(1, int(total_pages))
        estimated_chars_raw = getattr(profile, "estimated_chars", 0)

        if isinstance(estimated_chars_raw, (int, float)) and float(estimated_chars_raw) > 0:
            text_tokens = int(math.ceil(float(estimated_chars_raw) / 4.0))
            return text_tokens + (safe_pages * 500)

        return safe_pages * 1500

    def _call_openrouter(self, page_payloads: list[dict[str, Any]]) -> dict[str, Any]:
        httpx_module = self._load_httpx()
        if httpx_module is None:
            raise ValueError("httpx is required for StrategyC.")

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            return {"pages": []}

        # FORCE the correct model (ignores any wrong env var)
        model_name = "anthropic/claude-3.5-sonnet"
        print(f"DEBUG: Using model -> {model_name}")   # <-- added for confirmation

        instruction = (
            "You are a multimodal document analyst. Analyze each page's text and optional image. "
            "Identify section titles/headers versus normal paragraphs to support hierarchical indexing. "
            "Extract tables as both structured rows and markdown. "
            "Return STRICT JSON only using this schema: "
            "{\"pages\":[{"
            "\"page_number\":int,"
            "\"page_width\":number,"
            "\"page_height\":number,"
            "\"blocks\":[{\"kind\":\"title|header|paragraph\",\"text\":str}],"
            "\"text\":str,"
            "\"tables\":[{\"title\":str|null,\"headers\":[str],\"rows\":[[str]]}],"
            "\"markdown_tables\":[str]"
            "}]}"
        )

        content_items: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        for payload in page_payloads:
            content_items.append(
                {
                    "type": "text",
                    "text": (
                        f"Page {payload['page_number']} "
                        f"(width={payload.get('page_width', 0)}, height={payload.get('page_height', 0)}) text:\n"
                        f"{payload.get('text', '')}"
                    ),
                }
            )
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

        with httpx_module.Client(timeout=120.0, verify=False) as client:
            response = client.post(self.OPENROUTER_URL, headers=headers, json=request_body)
            print(f"DEBUG: LLM Raw Response: {response.text}")
            response.raise_for_status()
            payload = response.json()

        message_content = self._extract_message_content(payload)
        parsed = self._coerce_json_payload(message_content)
        if parsed.get("pages"):
            return parsed

        repaired_content = self._repair_json_with_retry(
            raw_text=message_content,
            model_name=model_name,
            headers=headers,
            httpx_module=httpx_module,
        )
        return self._coerce_json_payload(repaired_content)
    def _extract_message_content(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    def _coerce_json_payload(self, raw_text: str) -> dict[str, Any]:
        import re as _re

        clean_text = (raw_text or "").replace("```json", "").replace("```", "").strip()

        parse_candidates = [clean_text]
        brace_match = _re.search(r"\{[\s\S]*\}", clean_text)
        if brace_match:
            parse_candidates.append(brace_match.group(0).strip())

        for candidate in parse_candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

        return {"pages": []}

    def _repair_json_with_retry(
        self,
        raw_text: str,
        model_name: str,
        headers: dict[str, str],
        httpx_module: Any,
    ) -> str:
        repair_prompt = (
            "Repair the following malformed output into strict valid JSON only. "
            "Do not add commentary. Keep the same schema and preserve content where possible.\n\n"
            f"MALFORMED_OUTPUT:\n{raw_text}"
        )

        try:
            request_body = {
                "model": model_name,
                "messages": [{"role": "user", "content": [{"type": "text", "text": repair_prompt}]}],
                "temperature": 0,
            }
            with httpx_module.Client(timeout=60.0) as client:
                response = client.post(self.OPENROUTER_URL, headers=headers, json=request_body)
                response.raise_for_status()
                payload = response.json()
            return self._extract_message_content(payload)
        except Exception:
            return raw_text

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
        page_payloads: list[dict[str, Any]],
    ) -> tuple[list[LDU], list[dict[str, Any]], list[PageIndexNode]]:
        pages_data = response_payload.get("pages", [])
        ldus: list[LDU] = []
        provenance_items: list[dict[str, Any]] = []
        index_nodes: list[PageIndexNode] = []
        page_dims = {
            int(payload.get("page_number", 0)): (
                float(payload.get("page_width", 0.0) or 0.0),
                float(payload.get("page_height", 0.0) or 0.0),
            )
            for payload in page_payloads
            if isinstance(payload, dict) and payload.get("page_number")
        }

        if not isinstance(pages_data, list):
            pages_data = []

        for page_item in pages_data:
            if not isinstance(page_item, dict):
                continue

            page_number = int(page_item.get("page_number", len(index_nodes) + 1))
            item_width = float(page_item.get("page_width", 0.0) or 0.0)
            item_height = float(page_item.get("page_height", 0.0) or 0.0)
            known_width, known_height = page_dims.get(page_number, (0.0, 0.0))
            page_width = item_width if item_width > 0 else known_width
            page_height = item_height if item_height > 0 else known_height
            page_text = str(page_item.get("text", "")).strip()
            page_bbox = BBox(
                x_min=0.0,
                y_min=0.0,
                x_max=page_width,
                y_max=page_height,
                page_number=page_number,
            )

            blocks = page_item.get("blocks", [])
            if isinstance(blocks, list):
                for block_idx, block in enumerate(blocks, start=1):
                    if not isinstance(block, dict):
                        continue

                    kind = str(block.get("kind", "paragraph")).strip().casefold()
                    text = str(block.get("text", "")).strip()
                    if not text:
                        continue

                    mapped_kind = "paragraph"
                    if kind == "title":
                        mapped_kind = "title"
                    elif kind in {"header", "section_header", "heading"}:
                        mapped_kind = "header"

                    block_hash = self._hash_content(f"{mapped_kind}:{text}")
                    block_uid = f"{doc_id}-p{page_number:03d}-blk{block_idx:03d}"
                    ldus.append(
                        LDU(
                            uid=block_uid,
                            unit_type=mapped_kind,
                            content=text,
                            content_hash=block_hash,
                            page_refs=[page_number],
                            bounding_box=page_bbox,
                            parent_section=f"page-{page_number}",
                            child_chunks=self.chunk_text(text),
                        )
                    )
                    provenance_items.append(
                        ProvenanceChain(
                            source_file=filename,
                            content_hash=block_hash,
                            bbox=page_bbox,
                            strategy_used=profile.selected_strategy.value,
                        ).model_dump()
                    )

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

            markdown_tables = page_item.get("markdown_tables", [])
            if isinstance(markdown_tables, list):
                for md_idx, markdown_table in enumerate(markdown_tables, start=1):
                    md_text = str(markdown_table or "").strip()
                    if not md_text:
                        continue

                    md_hash = self._hash_content(md_text)
                    md_uid = f"{doc_id}-p{page_number:03d}-mdtbl{md_idx:02d}"
                    ldus.append(
                        LDU(
                            uid=md_uid,
                            unit_type="table",
                            content=md_text,
                            content_hash=md_hash,
                            page_refs=[page_number],
                            bounding_box=page_bbox,
                            parent_section=f"page-{page_number}",
                            child_chunks=self.chunk_text(md_text),
                        )
                    )
                    provenance_items.append(
                        ProvenanceChain(
                            source_file=filename,
                            content_hash=md_hash,
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

    def _partial_output(
        self,
        filename: str,
        doc_id: str,
        profile: DocumentProfile,
        warnings: list[str],
        projected_cost: float | None = None,
        estimated_tokens: int | None = None,
    ) -> NormalizedOutput:
        metadata: dict[str, Any] = {
            "selected_strategy": profile.selected_strategy.value,
            "avg_confidence": 0.0,
            "provenance_chain": [],
            "warning": " | ".join(warnings) if warnings else "StrategyC returned partial output",
        }
        if isinstance(projected_cost, (int, float)):
            metadata["projected_cost"] = round(float(projected_cost), 6)
        if isinstance(estimated_tokens, int):
            metadata["estimated_tokens"] = estimated_tokens

        return NormalizedOutput(
            filename=filename,
            doc_id=doc_id,
            profile=profile,
            ldus=[],
            index=[PageIndexNode(title="Page 1", page_start=1, page_end=1, children=[])],
            metadata=metadata,
        )
