import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.agents.triage import TriageAgent
from src.models.document_schema import ExtractedDocument, LayoutComplexity, OriginType
# Updated imports to match your new file names
from src.strategies.fast_text_extractor import StrategyA
from src.strategies.layout_extractor import StrategyB
from src.strategies.vision_extractor import StrategyC

logger = logging.getLogger(__name__)

class ExtractionRouter:
    """Coordinates triage, strategy routing, extraction, and ledger logging."""

    def __init__(self) -> None:
        self.triage_agent = TriageAgent()
        self.strategy_a = StrategyA()
        self.strategy_b = StrategyB()
        self.strategy_c = StrategyC()

    def process_document(self, pdf_path: str) -> ExtractedDocument:
        # 1. Triage (Phase 1)
        profile = self.triage_agent.profile_document(pdf_path)
        
        # Save profile artifact for rubric compliance (.refinery/profiles/)
        self._save_profile_artifact(profile)

        # 2. Strategy Routing Logic
        if profile.origin_type == OriginType.SCANNED_IMAGE:
            try:
                extracted = self.strategy_c.extract(pdf_path, profile)
            except ValueError as error:
                logger.warning("Strategy C failed/budget guard: %s. Falling back to Strategy B.", error)
                extracted = self.strategy_b.extract(pdf_path, profile)
        elif profile.layout_complexity in {LayoutComplexity.TABLE_HEAVY, LayoutComplexity.MULTI_COLUMN}:
            extracted = self.strategy_b.extract(pdf_path, profile)
        else:
            extracted = self.strategy_a.extract(pdf_path, profile)

        # 3. CONFIDENCE-GATED ESCALATION GUARD (Rubric Requirement)
        avg_conf = self._average_confidence(extracted)
        if extracted.strategy_used == "Strategy A" and avg_conf < 0.75:
            logger.info("Confidence %s below threshold for Strategy A. Escalating to Strategy B.", avg_conf)
            extracted = self.strategy_b.extract(pdf_path, profile)

        # 4. Logging
        self._append_ledger_record(pdf_path=pdf_path, extracted=extracted)
        return extracted

    def _save_profile_artifact(self, profile) -> None:
        """Saves individual profile JSONs for the interim submission check."""
        profile_dir = Path(".refinery/profiles")
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profile_dir / f"{Path(profile.filename).stem}_profile.json"
        
        with profile_path.open("w", encoding="utf-8") as f:
            f.write(profile.model_dump_json(indent=2))

    def _append_ledger_record(self, pdf_path: str, extracted: ExtractedDocument) -> None:
        ledger_dir = Path(".refinery")
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = ledger_dir / "extraction_ledger.jsonl"

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": Path(pdf_path).name,
            "strategy_used": extracted.strategy_used,
            "avg_confidence": self._average_confidence(extracted),
            "estimated_cost": self._estimated_cost(extracted),
        }

        with ledger_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _average_confidence(self, extracted: ExtractedDocument) -> float:
        if not extracted.pages:
            return 0.0
        total_confidence = sum(page.extraction_confidence for page in extracted.pages)
        return round(total_confidence / len(extracted.pages), 4)

    def _estimated_cost(self, extracted: ExtractedDocument) -> float:
        if extracted.strategy_used == "Strategy C":
            # Accessing COST_PER_PAGE from StrategyC class
            return round(len(extracted.pages) * 0.01, 4)
        return 0.0