import json
import logging
import os
import re
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        os.makedirs("logs", exist_ok=True)
        self.config = config or load_config(config_path)
        self.triage_agent = TriageAgent(config=self.config, config_path=config_path)
        self.strategy_a = StrategyA(config=self.config)
        self.strategy_b = StrategyB(config=self.config)
        self.strategy_c = StrategyC(config=self.config)
        self.ledger = ExtractionLedger("logs/extraction_ledger.jsonl")
        self.escalation_thresholds = self._load_escalation_thresholds()

    def process_document(self, pdf_path: str) -> NormalizedOutput:
        escalation_history: list[dict[str, Any]] = []
        total_processing_time = 0.0

        try:
            profile = self.triage_agent.profile_document(pdf_path)
            self._save_profile_artifact(profile)
            logger.info("Triage completed for %s | selected=%s", profile.filename, profile.selected_strategy.value)
        except BaseException as triage_error:
            logger.exception("Triage failed for %s. Returning partial output.", pdf_path)
            fallback_profile = self._fallback_profile(pdf_path)
            extracted = self._partial_output_from_error(
                profile=fallback_profile,
                error_message=f"Triage failed: {triage_error}",
                escalation_history=escalation_history,
            )
            extracted.metadata["processing_time"] = 0.0
            self._save_extraction_artifact(extracted)
            self._append_ledger_record(pdf_path=pdf_path, extracted=extracted)
            return extracted

        selected_strategy = profile.selected_strategy

        # Initial strategy execution based on triage decision
        extracted, current_processing_time = self._run_strategy(selected_strategy, pdf_path, profile, escalation_history)
        total_processing_time += current_processing_time
        if self._is_budget_exceeded_output(extracted):
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
