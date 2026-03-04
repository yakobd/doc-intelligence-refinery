import base64
import importlib
import io
import json
import os
from pathlib import Path
from typing import Any

import pdfplumber

from src.models.document_schema import (
    DocumentProfile,
    ExtractedDocument,
    ExtractedPage,
    Table,
)
from src.strategies.base_strategy import BaseStrategy

class StrategyC(BaseStrategy):
    """Vision-assisted extraction strategy for scanned or difficult PDFs."""

    COST_PER_PAGE = 0.01
    MAX_BUDGET = 0.50
    MAX_PAGES_TO_PROCESS = 3
    DEFAULT_MODEL = "google/gemini-flash-1.5"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
        pdf_name = Path(pdf_path).name
        doc_id = Path(pdf_path).stem
        page_payloads: list[dict[str, Any]] = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            page_payloads = self._build_page_payloads(pdf_path=pdf_path, pdf=pdf)

        projected_cost = total_pages * self.COST_PER_PAGE
        if projected_cost > self.MAX_BUDGET:
            raise ValueError(
                f"Budget exceeded for {pdf_name}: projected ${projected_cost:.2f} > max ${self.MAX_BUDGET:.2f}"
            )

        response_payload = self._call_openrouter(page_payloads=page_payloads)
        pages = self._map_response_to_pages(response_payload=response_payload)

        return ExtractedDocument(
            filename=pdf_name,
            doc_id=doc_id,
            profile=profile,
            pages=pages,
            strategy_used="Strategy C",
        )

    def _build_page_payloads(self, pdf_path: str, pdf: pdfplumber.PDF) -> list[dict[str, Any]]:
        page_payloads: list[dict[str, Any]] = []
        pages_to_process = min(self.MAX_PAGES_TO_PROCESS, len(pdf.pages))

        for page_number in range(1, pages_to_process + 1):
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

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            # We'll return empty data so the orchestrator can still log the attempt
            return {"pages": []}

        model_name = os.getenv("OPENROUTER_MODEL", self.DEFAULT_MODEL)
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
        if not choices: return ""
        return choices[0].get("message", {}).get("content", "")

    def _coerce_json_payload(self, raw_text: str) -> dict[str, Any]:
        try:
            # Simple cleaning for common LLM markdown output
            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except:
            return {"pages": []}

    def _load_httpx(self) -> Any:
        try: return importlib.import_module("httpx")
        except: return None

    def _load_pdf2image_converter(self) -> Any:
        try:
            module = importlib.import_module("pdf2image")
            return getattr(module, "convert_from_path", None)
        except: return None

    def _map_response_to_pages(self, response_payload: dict[str, Any]) -> list[ExtractedPage]:
        pages_data = response_payload.get("pages", [])
        mapped_pages = []
        for item in pages_data:
            tables = [Table(headers=t.get("headers", []), rows=t.get("rows", []), title=t.get("title")) 
                      for t in item.get("tables", [])]
            mapped_pages.append(ExtractedPage(
                page_number=item.get("page_number", 1),
                text=item.get("text", ""),
                tables=tables,
                extraction_confidence=0.85
            ))
        return mapped_pages or [ExtractedPage(page_number=1, text="API Call Placeholder", tables=[], extraction_confidence=0.85)]