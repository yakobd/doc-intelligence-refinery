import json
import logging
import os
import re
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.agents.triage import TriageAgent
from src.models.document_schema import (
    DocumentProfile,
    LDU,
    LayoutComplexity,
    NormalizedOutput,
    OriginType,
    PageIndexNode,
    StrategyTier,
)
from src.strategies.fast_text_extractor import StrategyA
from src.strategies.layout_extractor import StrategyB
from src.strategies.vision_extractor import StrategyC
from src.utils.config_loader import get_extraction_config, get_router_config, load_config

logger = logging.getLogger(__name__)


class ExtractionLedger:
    """Append-only JSONL ledger for extraction outcomes."""

    def __init__(self, ledger_path: str = "logs/extraction_ledger.jsonl") -> None:
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def log_extraction(self, record: dict[str, Any]) -> None:
        with self.ledger_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


class ExtractionRouter:
    """Coordinates triage, strategy routing, extraction, escalation, and ledger logging."""

    def __init__(self, config: dict[str, Any] | None = None, config_path: str = "rubric/extraction_rules.yaml") -> None:
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(dotenv_path=project_root / ".env")
        logs_dir = project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = logs_dir / "extraction_ledger.csv"
        self.config = config or load_config(config_path)
        self.triage_agent = TriageAgent(config=self.config, config_path=config_path)
        self.strategy_a = StrategyA(config=self.config)
        self.strategy_b = StrategyB(config=self.config)
        self.strategy_c = StrategyC(config=self.config)
        self.ledger = ExtractionLedger(str(self.ledger_path))
        self.escalation_thresholds = self._load_escalation_thresholds()

    def process_document(self, pdf_path: str) -> NormalizedOutput:
        print(f"[process_document] Starting extraction for: {pdf_path}")
        escalation_history: list[dict[str, Any]] = []
        total_processing_time = 0.0

        print("[process_document] Attempting LLM extraction initialization...")
        llm_ready = False
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not (google_api_key or "").strip():
                raise ValueError("GOOGLE_API_KEY is not set")

            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
            while model_name.startswith("models/"):
                model_name = model_name[len("models/") :]

            _ = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0,
                disable_streaming=True,
            )
            llm_ready = True
            print("[process_document] LLM initialization successful.")
        except Exception as llm_error:
            print(f"[process_document] LLM initialization failed: {llm_error}")
            print("[process_document] Falling back to PyMuPDF extraction path.")

        if not llm_ready:
            fallback_profile = self._fallback_profile(pdf_path)
            return self._fallback_ldus_from_pymupdf(
                pdf_path=pdf_path,
                profile=fallback_profile,
                warning_message="LLM initialization failed; used PyMuPDF fallback extraction.",
            )

        try:
            print("[process_document] Running triage...")
            profile = self.triage_agent.profile_document(pdf_path)
            self._save_profile_artifact(profile)
            print(
                "[process_document] Triage complete | "
                f"strategy={profile.selected_strategy.value} | pages={profile.pages}"
            )
            logger.info("Triage completed for %s | selected=%s", profile.filename, profile.selected_strategy.value)
        except BaseException as triage_error:
            logger.exception("Triage failed for %s. Returning partial output.", pdf_path)
            fallback_profile = self._fallback_profile(pdf_path)
            print(f"[process_document] Triage failed: {triage_error}")
            return self._fallback_ldus_from_pymupdf(
                pdf_path=pdf_path,
                profile=fallback_profile,
                warning_message=f"Triage failed: {triage_error}. Used PyMuPDF fallback extraction.",
            )

        selected_strategy = profile.selected_strategy

        # Initial strategy execution based on triage decision
        print(f"[process_document] Executing primary strategy: {selected_strategy.value}")
        extracted, current_processing_time = self._run_strategy(selected_strategy, pdf_path, profile, escalation_history)
        total_processing_time += current_processing_time
        print(
            "[process_document] Primary strategy finished | "
            f"ldus={len(extracted.ldus)} | processing_time={current_processing_time}s"
        )
        if self._is_budget_exceeded_output(extracted):
            print("[process_document] Budget guard triggered; returning budget-exceeded output.")
            extracted.profile = extracted.profile.model_copy(update={"selected_strategy": StrategyTier.STRATEGY_C})
            extracted.metadata["selected_strategy"] = StrategyTier.STRATEGY_C.value
            extracted.metadata["escalation_history"] = escalation_history
            extracted.metadata["processing_time"] = round(total_processing_time, 4)
            extracted = self._ensure_recursive_index(extracted)
            self._save_extraction_artifact(extracted)
            self._append_ledger_record(pdf_path=pdf_path, extracted=extracted)
            return extracted

        # Multi-level escalation flow (A -> B -> C)
        if selected_strategy == StrategyTier.STRATEGY_A:
            confidence_a = self._average_confidence(extracted)
            processing_time_a = current_processing_time
            action_a = "kept"
            if confidence_a < float(self.escalation_thresholds["min_confidence_a"]):
                action_a = "escalated"
                print(
                    "[process_document] Escalating A -> B | "
                    f"confidence={confidence_a:.4f} threshold={float(self.escalation_thresholds['min_confidence_a']):.4f}"
                )
                logger.info(
                    "Escalation decision for %s: FASTTEXT confidence %.4f < %.4f, escalating to LAYOUT",
                    profile.filename,
                    confidence_a,
                    float(self.escalation_thresholds["min_confidence_a"]),
                )
                extracted, current_processing_time = self._run_strategy(
                    StrategyTier.STRATEGY_B,
                    pdf_path,
                    profile,
                    escalation_history,
                )
                total_processing_time += current_processing_time
                if self._is_budget_exceeded_output(extracted):
                    print("[process_document] Budget guard triggered during escalation to B.")
                    extracted.profile = extracted.profile.model_copy(update={"selected_strategy": StrategyTier.STRATEGY_C})
                    extracted.metadata["selected_strategy"] = StrategyTier.STRATEGY_C.value
                    extracted.metadata["escalation_history"] = escalation_history
                    extracted.metadata["processing_time"] = round(total_processing_time, 4)
                    extracted = self._ensure_recursive_index(extracted)
                    self._save_extraction_artifact(extracted)
                    self._append_ledger_record(pdf_path=pdf_path, extracted=extracted)
                    return extracted
                selected_strategy = StrategyTier.STRATEGY_B
            escalation_history.append(
                {
                    "strategy": StrategyTier.STRATEGY_A.value,
                    "confidence": confidence_a,
                    "action": action_a,
                    "processing_time": round(processing_time_a, 4),
                }
            )

        if selected_strategy == StrategyTier.STRATEGY_B:
            confidence_b = self._average_confidence(extracted)
            processing_time_b = current_processing_time
            action_b = "kept"
            if confidence_b < float(self.escalation_thresholds["min_confidence_b"]):
                action_b = "escalated"
                print(
                    "[process_document] Escalating B -> C | "
                    f"confidence={confidence_b:.4f} threshold={float(self.escalation_thresholds['min_confidence_b']):.4f}"
                )
                logger.info(
                    "Escalation decision for %s: LAYOUT confidence %.4f < %.4f, escalating to VISION",
                    profile.filename,
                    confidence_b,
                    float(self.escalation_thresholds["min_confidence_b"]),
                )
                extracted, current_processing_time = self._run_strategy(
                    StrategyTier.STRATEGY_C,
                    pdf_path,
                    profile,
                    escalation_history,
                )
                total_processing_time += current_processing_time
                if self._is_budget_exceeded_output(extracted):
                    print("[process_document] Budget guard triggered during escalation to C.")
                    extracted.profile = extracted.profile.model_copy(update={"selected_strategy": StrategyTier.STRATEGY_C})
                    extracted.metadata["selected_strategy"] = StrategyTier.STRATEGY_C.value
                    extracted.metadata["escalation_history"] = escalation_history
                    extracted.metadata["processing_time"] = round(total_processing_time, 4)
                    extracted = self._ensure_recursive_index(extracted)
                    self._save_extraction_artifact(extracted)
                    self._append_ledger_record(pdf_path=pdf_path, extracted=extracted)
                    return extracted
                selected_strategy = StrategyTier.STRATEGY_C
            escalation_history.append(
                {
                    "strategy": StrategyTier.STRATEGY_B.value,
                    "confidence": confidence_b,
                    "action": action_b,
                    "processing_time": round(processing_time_b, 4),
                }
            )

        if selected_strategy == StrategyTier.STRATEGY_C:
            confidence_c = self._average_confidence(extracted)
            processing_time_c = current_processing_time
            requires_human_review = confidence_c < float(self.escalation_thresholds["min_confidence_final"])
            action_c = "human_review" if requires_human_review else "kept"
            escalation_history.append(
                {
                    "strategy": StrategyTier.STRATEGY_C.value,
                    "confidence": confidence_c,
                    "action": action_c,
                    "processing_time": round(processing_time_c, 4),
                }
            )
            logger.info(
                "Final Strategy C decision for %s: confidence=%.4f, threshold=%.4f, requires_human_review=%s",
                profile.filename,
                confidence_c,
                float(self.escalation_thresholds["min_confidence_final"]),
                requires_human_review,
            )
            extracted.metadata["requires_human_review"] = requires_human_review

        # Keep profile and metadata in sync with final strategy
        extracted.profile = extracted.profile.model_copy(update={"selected_strategy": selected_strategy})
        extracted.metadata["selected_strategy"] = selected_strategy.value
        extracted.metadata["escalation_history"] = escalation_history
        extracted.metadata["processing_time"] = round(total_processing_time, 4)

        extracted = self._refine_ldu_granularity(extracted)
        extracted = self._ensure_recursive_index(extracted)

        if not extracted.ldus:
            print("[process_document] Strategy pipeline returned zero LDUs; using PyMuPDF fallback.")
            return self._fallback_ldus_from_pymupdf(
                pdf_path=pdf_path,
                profile=profile,
                warning_message="Primary extraction produced no LDUs; used PyMuPDF fallback extraction.",
            )

        total_text_length = sum(len((ldu.content or "").strip()) for ldu in extracted.ldus)
        print(f"[process_document] Raw text length found: {total_text_length}")
        print(f"[process_document] Final LDU count: {len(extracted.ldus)}")

        self._save_extraction_artifact(extracted)
        self._append_ledger_record(pdf_path=pdf_path, extracted=extracted)
        return extracted

    def _fallback_ldus_from_pymupdf(
        self,
        pdf_path: str,
        profile: DocumentProfile,
        warning_message: str,
    ) -> NormalizedOutput:
        print("[process_document] Entering PyMuPDF fallback extraction...")
        try:
            import fitz  # PyMuPDF
        except Exception as fitz_error:
            logger.exception("PyMuPDF import failed during fallback for %s", pdf_path)
            print(f"[process_document] PyMuPDF import failed: {fitz_error}")
            extracted = self._partial_output_from_error(
                profile=profile,
                error_message=f"{warning_message} PyMuPDF import failed: {fitz_error}",
                escalation_history=[],
            )
            extracted.metadata["processing_time"] = 0.0
            self._save_extraction_artifact(extracted)
            self._append_ledger_record(pdf_path=pdf_path, extracted=extracted)
            return extracted

        filename = Path(pdf_path).name
        doc_id = Path(pdf_path).stem
        ldus: list[LDU] = []
        index_nodes: list[PageIndexNode] = []
        provenance_items: list[dict[str, Any]] = []
        total_text_length = 0

        try:
            with fitz.open(pdf_path) as doc:
                for page_idx in range(len(doc)):
                    page_number = page_idx + 1
                    page = doc[page_idx]
                    page_text = (page.get_text("text") or "").strip()

                    if not page_text:
                        print(
                            "[process_document] Fallback page text empty from get_text('text') | "
                            f"page={page_number}. Trying blocks..."
                        )
                        try:
                            blocks = page.get_text("blocks") or []
                            block_texts: list[str] = []
                            for block in blocks:
                                if isinstance(block, (list, tuple)) and len(block) >= 5:
                                    text_part = str(block[4] or "").strip()
                                    if text_part:
                                        block_texts.append(text_part)
                                elif isinstance(block, dict):
                                    text_part = str(block.get("text", "")).strip()
                                    if text_part:
                                        block_texts.append(text_part)
                            page_text = "\n".join(block_texts).strip()
                        except Exception as blocks_error:
                            print(
                                "[process_document] get_text('blocks') failed | "
                                f"page={page_number} error={blocks_error}"
                            )

                    if not page_text:
                        try:
                            pix = page.get_pixmap(alpha=False)
                            if pix.width > 0 and pix.height > 0:
                                print("Page appears to be an image.")
                                print(
                                    "[process_document] No extractable text found; "
                                    f"page={page_number} pixmap={pix.width}x{pix.height}"
                                )
                        except Exception as pix_error:
                            print(
                                "[process_document] get_pixmap() failed during image check | "
                                f"page={page_number} error={pix_error}"
                            )

                    # Normalize whitespace to keep chunk quality and hash stability.
                    page_text = re.sub(r"\s+", " ", (page_text or "")).strip()

                    page_text_length = len(page_text)
                    total_text_length += page_text_length
                    print(
                        "[process_document] Fallback page read | "
                        f"page={page_number} text_len={page_text_length}"
                    )

                    if not page_text:
                        index_nodes.append(
                            PageIndexNode(
                                title=f"Page {page_number}",
                                page_start=page_number,
                                page_end=page_number,
                                children=[],
                            )
                        )
                        continue

                    chunk_size = 1000
                    page_chunks = [
                        page_text[i : i + chunk_size].strip()
                        for i in range(0, len(page_text), chunk_size)
                        if page_text[i : i + chunk_size].strip()
                    ]

                    for chunk_idx, chunk_text in enumerate(page_chunks, start=1):
                        normalized_chunk = re.sub(r"\s+", " ", (chunk_text or "")).strip()
                        if not normalized_chunk:
                            continue

                        # Include text + page context for uniqueness across repeated values.
                        chunk_hash = hashlib.sha256(
                            f"{normalized_chunk}|{filename}|{page_number}|{chunk_idx}".encode("utf-8")
                        ).hexdigest()
                        bbox = [0.0, 0.0, 0.0, 0.0]

                        ldus.append(
                            LDU(
                                uid=f"{doc_id}-p{page_number:03d}-fallback-{chunk_idx:03d}",
                                chunk_type="paragraph",
                                content=normalized_chunk,
                                content_hash=chunk_hash,
                                page_refs=[page_number],
                                bounding_box=bbox,
                                parent_section=f"page-{page_number}",
                                token_count=len(normalized_chunk.split()),
                                child_chunks=[normalized_chunk],
                            )
                        )
                        provenance_items.append(
                            {
                                "document_name": filename,
                                "page_number": page_number,
                                "bbox": bbox,
                                "content_hash": chunk_hash,
                            }
                        )

                    index_nodes.append(
                        PageIndexNode(
                            title=f"Page {page_number}",
                            page_start=page_number,
                            page_end=page_number,
                            children=[],
                        )
                    )
        except Exception as fallback_error:
            logger.exception("PyMuPDF fallback failed for %s", pdf_path)
            print(f"[process_document] PyMuPDF fallback failed: {fallback_error}")
            extracted = self._partial_output_from_error(
                profile=profile,
                error_message=f"{warning_message} PyMuPDF extraction failed: {fallback_error}",
                escalation_history=[],
            )
            extracted.metadata["processing_time"] = 0.0
            self._save_extraction_artifact(extracted)
            self._append_ledger_record(pdf_path=pdf_path, extracted=extracted)
            return extracted

        print(f"[process_document] Raw text length found: {total_text_length}")
        print(f"[process_document] Fallback LDU count: {len(ldus)}")

        if not ldus and total_text_length > 0:
            # Safety net: if text exists but chunking produced no LDUs.
            chunk_hash = hashlib.sha256(f"{doc_id}|fallback".encode("utf-8")).hexdigest()
            ldus.append(
                LDU(
                    uid=f"{doc_id}-fallback-001",
                    chunk_type="paragraph",
                    content="Recovered text was detected but could not be chunked.",
                    content_hash=chunk_hash,
                    page_refs=[1],
                    bounding_box=[0.0, 0.0, 0.0, 0.0],
                    parent_section="page-1",
                    token_count=9,
                    child_chunks=["Recovered text was detected but could not be chunked."],
                )
            )

        extracted = NormalizedOutput(
            filename=filename,
            doc_id=doc_id,
            profile=profile,
            ldus=ldus,
            index=index_nodes if index_nodes else [PageIndexNode(title="Page 1", page_start=1, page_end=1, children=[])],
            metadata={
                "selected_strategy": profile.selected_strategy.value,
                "avg_confidence": 0.6,
                "provenance_chain": provenance_items,
                "warning": warning_message,
                "requires_human_review": False,
                "escalation_history": [],
                "processing_time": 0.0,
                "source_filename": filename,
                "fallback_pages_processed": len(index_nodes),
            },
        )
        self._save_extraction_artifact(extracted)
        self._append_ledger_record(pdf_path=pdf_path, extracted=extracted)
        return extracted

    def _run_strategy(
        self,
        strategy: StrategyTier,
        pdf_path: str,
        profile: DocumentProfile,
        escalation_history: list[dict[str, Any]],
    ) -> tuple[NormalizedOutput, float]:
        start_time = time.perf_counter()
        logger.info("Routing decision: executing %s for %s", strategy.value, profile.filename)
        try:
            if strategy == StrategyTier.STRATEGY_A:
                extracted = self.strategy_a.extract(pdf_path, profile)
            elif strategy == StrategyTier.STRATEGY_B:
                extracted = self.strategy_b.extract(pdf_path, profile)
            else:
                extracted = self.strategy_c.extract(pdf_path, profile)

            processing_time = round(time.perf_counter() - start_time, 4)
            extracted.metadata["processing_time"] = processing_time
            return extracted, processing_time
        except ValueError as strategy_error:
            processing_time = round(time.perf_counter() - start_time, 4)
            if self._is_budget_error(strategy_error):
                projected_cost = self._extract_projected_cost(str(strategy_error))
                warning = (
                    f"Budget guard blocked Strategy C for {profile.filename}: {strategy_error}. "
                    "Document skipped due to cost constraints."
                )
                logger.warning(warning)
                escalation_history.append(
                    {
                        "strategy": strategy.value,
                        "confidence": 0.0,
                        "action": "budget_exceeded",
                        "error": str(strategy_error),
                        "processing_time": processing_time,
                    }
                )
                extracted = self._budget_exceeded_output(
                    profile=profile,
                    warning_message=warning,
                    projected_cost=projected_cost,
                    escalation_history=escalation_history,
                )
                extracted.metadata["processing_time"] = processing_time
                return extracted, processing_time

            logger.exception("Strategy %s failed for %s", strategy.value, profile.filename)
            escalation_history.append(
                {
                    "strategy": strategy.value,
                    "confidence": 0.0,
                    "action": "failed",
                    "error": str(strategy_error),
                    "processing_time": processing_time,
                }
            )
            extracted = self._partial_output_from_error(
                profile=profile,
                error_message=f"Strategy {strategy.value} failed: {strategy_error}",
                escalation_history=escalation_history,
            )
            extracted.metadata["processing_time"] = processing_time
            return extracted, processing_time
        except BaseException as strategy_error:
            logger.exception("Strategy %s failed for %s", strategy.value, profile.filename)
            processing_time = round(time.perf_counter() - start_time, 4)
            escalation_history.append(
                {
                    "strategy": strategy.value,
                    "confidence": 0.0,
                    "action": "failed",
                    "error": str(strategy_error),
                    "processing_time": processing_time,
                }
            )
            extracted = self._partial_output_from_error(
                profile=profile,
                error_message=f"Strategy {strategy.value} failed: {strategy_error}",
                escalation_history=escalation_history,
            )
            extracted.metadata["processing_time"] = processing_time
            return extracted, processing_time

    def _is_budget_error(self, error: ValueError) -> bool:
        message = str(error).lower()
        return "budget exceeded" in message

    def _extract_projected_cost(self, error_message: str) -> float:
        match = re.search(r"projected\s*\$\s*([0-9]+(?:\.[0-9]+)?)", error_message, flags=re.IGNORECASE)
        if not match:
            return 0.0
        try:
            return round(float(match.group(1)), 6)
        except ValueError:
            return 0.0

    def _is_budget_exceeded_output(self, extracted: NormalizedOutput) -> bool:
        return extracted.metadata.get("status") == "BUDGET_EXCEEDED"

    def _budget_exceeded_output(
        self,
        profile: DocumentProfile,
        warning_message: str,
        projected_cost: float,
        escalation_history: list[dict[str, Any]],
    ) -> NormalizedOutput:
        return NormalizedOutput(
            filename=profile.filename,
            doc_id=Path(profile.filename).stem,
            profile=profile,
            ldus=[],
            index=[PageIndexNode(title="Page 1", page_start=1, page_end=1, children=[])],
            metadata={
                "status": "BUDGET_EXCEEDED",
                "selected_strategy": StrategyTier.STRATEGY_C.value,
                "avg_confidence": 0.0,
                "projected_cost": projected_cost,
                "provenance_chain": [],
                "warning": warning_message,
                "requires_human_review": True,
                "escalation_history": escalation_history,
            },
        )

    def _load_escalation_thresholds(self) -> dict[str, float]:
        default_thresholds = {
            "min_confidence_a": 0.8,
            "min_confidence_b": 0.7,
            "min_confidence_final": 0.65,
        }

        router_cfg = get_router_config(self.config)
        escalation_cfg = router_cfg.get("escalation_thresholds", {}) if isinstance(router_cfg, dict) else {}
        if not isinstance(escalation_cfg, dict):
            extraction_cfg = get_extraction_config(self.config)
            escalation_cfg = extraction_cfg.get("escalation_thresholds", {}) if isinstance(extraction_cfg, dict) else {}

        if not isinstance(escalation_cfg, dict):
            return default_thresholds

        if "min_confidence_c" in escalation_cfg and "min_confidence_final" not in escalation_cfg:
            escalation_cfg["min_confidence_final"] = escalation_cfg["min_confidence_c"]

        thresholds = default_thresholds.copy()
        for key in thresholds:
            value = escalation_cfg.get(key)
            if isinstance(value, (int, float)):
                thresholds[key] = float(value)

        return thresholds

    def _fallback_profile(self, pdf_path: str) -> DocumentProfile:
        return DocumentProfile(
            filename=Path(pdf_path).name,
            origin_type=OriginType.MIXED,
            layout_complexity=LayoutComplexity.MULTI_COLUMN,
            selected_strategy=StrategyTier.STRATEGY_C,
            confidence_score=0.0,
            estimated_cost=0.0,
            pages=1,
            language="en",
            domain_hint="general",
        )

    def _partial_output_from_error(
        self,
        profile: DocumentProfile,
        error_message: str,
        escalation_history: list[dict[str, Any]],
    ) -> NormalizedOutput:
        return NormalizedOutput(
            filename=profile.filename,
            doc_id=Path(profile.filename).stem,
            profile=profile,
            ldus=[],
            index=[PageIndexNode(title="Page 1", page_start=1, page_end=1, children=[])],
            metadata={
                "selected_strategy": profile.selected_strategy.value,
                "avg_confidence": 0.0,
                "provenance_chain": [],
                "warning": error_message,
                "requires_human_review": True,
                "escalation_history": escalation_history,
            },
        )

    def _save_profile_artifact(self, profile: DocumentProfile) -> None:
        profile_dir = Path(".refinery/profiles")
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profile_dir / f"{Path(profile.filename).stem}_profile.json"

        with profile_path.open("w", encoding="utf-8") as file:
            file.write(profile.model_dump_json(indent=2))

    def _append_ledger_record(self, pdf_path: str, extracted: NormalizedOutput) -> None:
        selected_strategy = extracted.metadata.get("selected_strategy", extracted.profile.selected_strategy.value)
        final_cost = self._estimated_cost(extracted)
        status = str(extracted.metadata.get("status", "SUCCESS"))
        if status == "BUDGET_EXCEEDED":
            cost_status = "BUDGET_EXCEEDED"
        elif status == "SUCCESS":
            cost_status = "WITHIN_BUDGET"
        else:
            cost_status = "FAILED"

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": Path(pdf_path).name,
            "strategy_used": selected_strategy,
            "confidence_score": self._average_confidence(extracted),
            "final_cost": final_cost,
            "cost_estimate": final_cost,
            "cost_status": cost_status,
            "processing_time": round(float(extracted.metadata.get("processing_time", 0.0)), 4),
            "status": status,
            "warning": extracted.metadata.get("warning", ""),
        }
        self.ledger.log_extraction(record)

    def _save_extraction_artifact(self, extracted: NormalizedOutput) -> None:
        extraction_dir = Path(".refinery/extractions")
        extraction_dir.mkdir(parents=True, exist_ok=True)
        extraction_path = extraction_dir / f"{Path(extracted.filename).stem}_extracted.json"

        with extraction_path.open("w", encoding="utf-8") as file:
            file.write(extracted.model_dump_json(indent=2))

    def _average_confidence(self, extracted: NormalizedOutput) -> float:
        metadata_confidence = extracted.metadata.get("avg_confidence")
        if isinstance(metadata_confidence, (int, float)):
            return round(float(metadata_confidence), 4)

        if not extracted.ldus:
            return 0.0

        router_cfg = get_router_config(self.config)
        fallback_confidence = 0.0
        if isinstance(router_cfg, dict):
            value = router_cfg.get("default_fallback_confidence")
            if isinstance(value, (int, float)):
                fallback_confidence = float(value)

        return round(fallback_confidence, 4)

    def _estimated_cost(self, extracted: NormalizedOutput) -> float:
        metadata_projected_cost = extracted.metadata.get("projected_cost")
        if isinstance(metadata_projected_cost, (int, float)):
            return round(float(metadata_projected_cost), 6)

        selected_strategy = extracted.metadata.get("selected_strategy", extracted.profile.selected_strategy.value)
        if selected_strategy == StrategyTier.STRATEGY_C.value:
            return 0.0
        return 0.0

    def _refine_ldu_granularity(self, extracted: NormalizedOutput) -> NormalizedOutput:
        refined_ldus: list[LDU] = []

        for ldu in extracted.ldus:
            # Keep table-like structures intact to avoid breaking row semantics.
            if (ldu.unit_type or "").lower() == "table":
                refined_ldus.append(ldu)
                continue

            raw_lines = (ldu.content or "").split("\n")
            if not raw_lines:
                continue

            segments: list[tuple[str, str]] = []
            current_body_lines: list[str] = []

            for raw_line in raw_lines:
                line = raw_line.strip()

                if not line:
                    if current_body_lines:
                        segments.append(("paragraph", " ".join(current_body_lines).strip()))
                        current_body_lines = []
                    continue

                if self._is_header_line(line):
                    if current_body_lines:
                        segments.append(("paragraph", " ".join(current_body_lines).strip()))
                        current_body_lines = []
                    segments.append(("header", line))
                    continue

                current_body_lines.append(line)

            if current_body_lines:
                segments.append(("paragraph", " ".join(current_body_lines).strip()))

            segment_counter = 0
            for segment_unit_type, segment_text in segments:
                segment = (segment_text or "").strip()
                if not segment:
                    continue

                segment_counter += 1
                segment_hash = hashlib.md5(segment.encode("utf-8")).hexdigest()
                refined_ldus.append(
                    LDU(
                        uid=f"{ldu.uid}-seg{segment_counter:02d}",
                        unit_type=segment_unit_type,
                        content=segment,
                        content_hash=segment_hash,
                        page_refs=list(ldu.page_refs),
                        bounding_box=ldu.bounding_box,
                        parent_section=ldu.parent_section,
                        child_chunks=[segment],
                        chunks=list(ldu.chunks),
                    )
                )

        extracted.ldus = refined_ldus
        return extracted

    def _is_header_line(self, line: str) -> bool:
        candidate = (line or "").strip()
        if not candidate:
            return False

        if len(candidate) >= 80 or candidate.endswith("."):
            return False

        if re.match(r"^\d", candidate):
            return True

        if candidate.isupper():
            return True

        if candidate.istitle():
            return True

        return False

    def _ensure_recursive_index(self, extracted: NormalizedOutput) -> NormalizedOutput:
        if extracted.index:
            normalized_nodes = [
                PageIndexNode(
                    title=node.title,
                    page_start=node.page_start,
                    page_end=node.page_end,
                    children=node.children or [],
                )
                for node in extracted.index
            ]
            extracted.index = normalized_nodes
            return extracted

        page_refs = sorted({page for ldu in extracted.ldus for page in ldu.page_refs})
        if not page_refs:
            extracted.index = [PageIndexNode(title="Page 1", page_start=1, page_end=1, children=[])]
            return extracted

        page_titles: dict[int, str] = {}
        for ldu in extracted.ldus:
            if (ldu.unit_type or "").lower() != "header":
                continue

            header_text = (ldu.content or "").strip().splitlines()[0] if ldu.content else ""
            if not header_text:
                continue

            for page in ldu.page_refs:
                if page not in page_titles:
                    page_titles[page] = header_text

        extracted.index = [
            PageIndexNode(
                title=page_titles.get(page_number, f"Page {page_number}"),
                page_start=page_number,
                page_end=page_number,
                children=[],
            )
            for page_number in page_refs
        ]
        return extracted
