import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents.triage import TriageAgent
from src.models.document_schema import (
    DocumentProfile,
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


class ExtractionRouter:
    """Coordinates triage, strategy routing, extraction, escalation, and ledger logging."""

    def __init__(self, config: dict[str, Any] | None = None, config_path: str = "rubric/extraction_rules.yaml") -> None:
        self.config = config or load_config(config_path)
        self.triage_agent = TriageAgent(config=self.config, config_path=config_path)
        self.strategy_a = StrategyA(config=self.config)
        self.strategy_b = StrategyB(config=self.config)
        self.strategy_c = StrategyC(config=self.config)
        self.escalation_thresholds = self._load_escalation_thresholds()

    def process_document(self, pdf_path: str) -> NormalizedOutput:
        escalation_history: list[dict[str, Any]] = []

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
            self._save_extraction_artifact(extracted)
            self._append_ledger_record(pdf_path=pdf_path, extracted=extracted)
            return extracted

        selected_strategy = profile.selected_strategy

        # Initial strategy execution based on triage decision
        extracted = self._run_strategy(selected_strategy, pdf_path, profile, escalation_history)

        # Multi-level escalation flow (A -> B -> C)
        if selected_strategy == StrategyTier.STRATEGY_A:
            confidence_a = self._average_confidence(extracted)
            action_a = "kept"
            if confidence_a < float(self.escalation_thresholds["min_confidence_a"]):
                action_a = "escalated"
                logger.info(
                    "Escalation decision for %s: FASTTEXT confidence %.4f < %.4f, escalating to LAYOUT",
                    profile.filename,
                    confidence_a,
                    float(self.escalation_thresholds["min_confidence_a"]),
                )
                extracted = self._run_strategy(StrategyTier.STRATEGY_B, pdf_path, profile, escalation_history)
                selected_strategy = StrategyTier.STRATEGY_B
            escalation_history.append({"strategy": StrategyTier.STRATEGY_A.value, "confidence": confidence_a, "action": action_a})

        if selected_strategy == StrategyTier.STRATEGY_B:
            confidence_b = self._average_confidence(extracted)
            action_b = "kept"
            if confidence_b < float(self.escalation_thresholds["min_confidence_b"]):
                action_b = "escalated"
                logger.info(
                    "Escalation decision for %s: LAYOUT confidence %.4f < %.4f, escalating to VISION",
                    profile.filename,
                    confidence_b,
                    float(self.escalation_thresholds["min_confidence_b"]),
                )
                extracted = self._run_strategy(StrategyTier.STRATEGY_C, pdf_path, profile, escalation_history)
                selected_strategy = StrategyTier.STRATEGY_C
            escalation_history.append({"strategy": StrategyTier.STRATEGY_B.value, "confidence": confidence_b, "action": action_b})

        if selected_strategy == StrategyTier.STRATEGY_C:
            confidence_c = self._average_confidence(extracted)
            requires_human_review = confidence_c < float(self.escalation_thresholds["min_confidence_final"])
            action_c = "human_review" if requires_human_review else "kept"
            escalation_history.append({"strategy": StrategyTier.STRATEGY_C.value, "confidence": confidence_c, "action": action_c})
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
    ) -> NormalizedOutput:
        logger.info("Routing decision: executing %s for %s", strategy.value, profile.filename)
        try:
            if strategy == StrategyTier.STRATEGY_A:
                return self.strategy_a.extract(pdf_path, profile)
            if strategy == StrategyTier.STRATEGY_B:
                return self.strategy_b.extract(pdf_path, profile)
            return self.strategy_c.extract(pdf_path, profile)
        except BaseException as strategy_error:
            logger.exception("Strategy %s failed for %s", strategy.value, profile.filename)
            escalation_history.append(
                {
                    "strategy": strategy.value,
                    "confidence": 0.0,
                    "action": "failed",
                    "error": str(strategy_error),
                }
            )
            return self._partial_output_from_error(
                profile=profile,
                error_message=f"Strategy {strategy.value} failed: {strategy_error}",
                escalation_history=escalation_history,
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
        ledger_dir = Path(".refinery")
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = ledger_dir / "extraction_ledger.jsonl"

        selected_strategy = extracted.metadata.get("selected_strategy", extracted.profile.selected_strategy.value)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": Path(pdf_path).name,
            "strategy_used": selected_strategy,
            "avg_confidence": self._average_confidence(extracted),
            "estimated_cost": self._estimated_cost(extracted),
        }

        with ledger_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

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

        return 0.85

    def _estimated_cost(self, extracted: NormalizedOutput) -> float:
        selected_strategy = extracted.metadata.get("selected_strategy", extracted.profile.selected_strategy.value)
        if selected_strategy == StrategyTier.STRATEGY_C.value:
            unique_pages = {page for ldu in extracted.ldus for page in ldu.page_refs}
            pages_count = len(unique_pages) if unique_pages else max(1, extracted.profile.pages)
            return round(pages_count * StrategyC.COST_PER_PAGE, 4)
        return 0.0

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

        extracted.index = [
            PageIndexNode(
                title=f"Page {page_number}",
                page_start=page_number,
                page_end=page_number,
                children=[],
            )
            for page_number in page_refs
        ]
        return extracted
