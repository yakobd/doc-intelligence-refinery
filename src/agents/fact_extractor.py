import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import re
import sqlite3
import json
from pathlib import Path
from typing import Any, List

from langchain_google_genai import ChatGoogleGenerativeAI

from src.models.document_schema import ExtractedFact, LDU, ProvenanceChain



class FactTableExtractor:
    """Extracts high-confidence numerical facts and stores them in SQLite."""

    HIGH_CONFIDENCE_THRESHOLD = 0.85

    def __init__(self, db_path: str = ".refinery/facts.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.llm: ChatGoogleGenerativeAI | None = None
        if self.google_api_key:
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
            while model_name.startswith("models/"):
                model_name = model_name[len("models/") :]
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0,
                disable_streaming=True,
            )
        self.initialize_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def initialize_db(self) -> None:
        """Create the fact_table if it does not already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fact_table (
                    fact_name TEXT NOT NULL,
                    value TEXT NOT NULL,
                    unit TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    document_name TEXT NOT NULL DEFAULT 'unknown',
                    bbox TEXT NOT NULL DEFAULT '[0.0, 0.0, 0.0, 0.0]'
                )
                """
            )

            # Lightweight migration path for older tables created before provenance fields.
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(fact_table)").fetchall()
            }
            if "document_name" not in existing_columns:
                conn.execute(
                    "ALTER TABLE fact_table ADD COLUMN document_name TEXT NOT NULL DEFAULT 'unknown'"
                )
            if "bbox" not in existing_columns:
                conn.execute(
                    "ALTER TABLE fact_table ADD COLUMN bbox TEXT NOT NULL DEFAULT '[0.0, 0.0, 0.0, 0.0]'"
                )
            conn.commit()

    def extract_facts_from_ldus(self, ldus: List[LDU]) -> List[ExtractedFact]:
        """
        Extract high-confidence numerical facts from LDUs and persist them.

        A prompt template for a fast/cheap model (for example Gemini Flash) is
        constructed for each LDU. The current implementation uses deterministic
        parsing so it can run offline in tests and local pipelines.
        """
        extracted_facts: list[ExtractedFact] = []
        rows_to_insert: list[tuple[str, str, str, int, str, str, str]] = []

        for ldu in ldus:
            text = (ldu.content or "").strip()
            if not text or not re.search(r"\d", text):
                continue

            page_number = int(ldu.page_refs[0]) if ldu.page_refs else 1
            bbox = list(ldu.bounding_box) if isinstance(ldu.bounding_box, list) else [0.0, 0.0, 0.0, 0.0]

            llm_candidates = self._extract_candidates_with_llm(text)
            candidates = llm_candidates if llm_candidates else self._extract_high_confidence_candidates(text)

            for candidate in candidates:
                fact_name = candidate["fact_name"]
                value = candidate["value"]
                unit = candidate["unit"]
                document_name = str(
                    getattr(ldu, "document_name", None)
                    or getattr(ldu, "filename", None)
                    or "unknown"
                )

                provenance = ProvenanceChain(
                    document_name=document_name,
                    page_number=page_number,
                    bbox=bbox,
                    content_hash=ldu.content_hash,
                )
                fact = ExtractedFact(
                    fact_name=fact_name,
                    value=value,
                    unit=unit,
                    provenance=provenance,
                )
                extracted_facts.append(fact)
                rows_to_insert.append(
                    (
                        fact.fact_name,
                        str(fact.value),
                        fact.unit,
                        fact.provenance.page_number,
                        fact.provenance.content_hash,
                        fact.provenance.document_name,
                        json.dumps(fact.provenance.bbox),
                    )
                )

        if rows_to_insert:
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO fact_table (fact_name, value, unit, page, content_hash, document_name, bbox)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows_to_insert,
                )
                conn.commit()

        return extracted_facts

    def _extract_candidates_with_llm(self, text: str) -> list[dict[str, Any]]:
        if self.llm is None:
            return []

        prompt = (
            "You are a precise numerical fact extractor. Extract only high-confidence numerical data "
            "from the passage. Return JSON array only (no markdown) with objects containing: "
            "fact_name (string), value (number or string for date), unit (string), confidence (0-1). "
            "Include only items with confidence >= 0.85 and focus on revenue, inflation rates, "
            "percentages, index values, totals, and dates.\n\n"
            f"Passage:\n{text[:2400]}"
        )

        try:
            response = self.llm.invoke(prompt)
            raw_text = str(getattr(response, "content", "")).strip()
            cleaned = self._strip_json_fences(raw_text)
            payload = json.loads(cleaned)
            if not isinstance(payload, list):
                return []
        except Exception:
            return []

        extracted: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in payload:
            if not isinstance(item, dict):
                continue

            confidence = item.get("confidence", 0)
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                continue

            if confidence_value < self.HIGH_CONFIDENCE_THRESHOLD:
                continue

            fact_name = str(item.get("fact_name", "")).strip()
            value = item.get("value", "")
            unit = str(item.get("unit", "")).strip()
            if not fact_name:
                continue

            value_as_text = str(value)
            if not re.search(r"\d", value_as_text) and unit.casefold() != "date":
                continue

            key = (fact_name.casefold(), value_as_text, unit.casefold())
            if key in seen:
                continue

            seen.add(key)
            extracted.append({"fact_name": fact_name, "value": value, "unit": unit})

        return extracted

    def _extract_high_confidence_candidates(self, text: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        # Pattern to catch "Label: Value" or "Label - Value"
        pattern = re.compile(r"(?P<name>[\w\s]{3,30})[:\-]\s*(?P<value>[\w\d\s\.]+)")

        for match in pattern.finditer(text):
            name = match.group("name").strip()
            val = match.group("value").strip()
            if len(val) > 0 and len(name) > 2:
                candidates.append(
                    {
                        "fact_name": name,
                        "value": val,
                        "unit": "text/number",
                        "confidence": 0.9,
                    }
                )
        return candidates

    def query_facts(self, sql_query: str) -> list[dict[str, Any]]:
        """Execute read-only SQL against fact_table for retrieval workflows."""
        query = (sql_query or "").strip()
        if not query:
            return []

        if not query.casefold().startswith("select"):
            raise ValueError("query_facts only allows SELECT statements")

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query).fetchall()
            return [dict(row) for row in rows]

    def verify_fact(self, claim_text: str, expected_hash: str | None = None) -> ProvenanceChain | str:
        """
        Audit Mode verification for factual claims.

        Returns a ProvenanceChain for an evidence-backed match, or 'unverifiable'
        when no exact numeric/hash anchored evidence is found.
        """
        claim = (claim_text or "").strip()
        if not claim:
            return "unverifiable"

        normalized_expected_hash = (expected_hash or "").strip()

        number_tokens = [token.replace(",", "") for token in re.findall(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?", claim)]
        keyword_tokens = [
            token
            for token in re.findall(r"[A-Za-z]{3,}", claim.casefold())
            if token not in {"the", "and", "for", "with", "that", "from", "this", "were", "was"}
        ]

        conditions: list[str] = []
        params: list[Any] = []
        for token in number_tokens:
            conditions.append("REPLACE(value, ',', '') LIKE ?")
            params.append(f"%{token}%")
        for token in keyword_tokens:
            conditions.append("LOWER(fact_name) LIKE ?")
            params.append(f"%{token}%")

        if not conditions:
            return "unverifiable"

        sql = (
            "SELECT fact_name, value, unit, page, content_hash, document_name, bbox "
            "FROM fact_table "
            f"WHERE {' OR '.join(conditions)}"
        )

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return "unverifiable"

        for row in rows:
            row_hash = str(row["content_hash"] or "")
            if normalized_expected_hash and row_hash != normalized_expected_hash:
                continue

            # Audit rule: for claims containing numeric values, require exact numeric evidence.
            if number_tokens:
                row_value_normalized = str(row["value"] or "").replace(",", "")
                if not any(self._numeric_equal(token, row_value_normalized) for token in number_tokens):
                    continue

            raw_bbox = row["bbox"]
            bbox = self._parse_bbox(raw_bbox)
            return ProvenanceChain(
                document_name=str(row["document_name"] or "unknown"),
                page_number=int(row["page"]),
                bbox=bbox,
                content_hash=row_hash,
            )

        return "unverifiable"

    def _parse_bbox(self, raw_bbox: Any) -> list[float]:
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            return [float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])]

        if isinstance(raw_bbox, str):
            try:
                decoded = json.loads(raw_bbox)
                if isinstance(decoded, list) and len(decoded) >= 4:
                    return [float(decoded[0]), float(decoded[1]), float(decoded[2]), float(decoded[3])]
            except json.JSONDecodeError:
                pass

        return [0.0, 0.0, 0.0, 0.0]

    def _numeric_equal(self, left: str, right: str) -> bool:
        try:
            return float(left) == float(right)
        except ValueError:
            return left == right

    def _strip_json_fences(self, value: str) -> str:
        text = (value or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        return text
