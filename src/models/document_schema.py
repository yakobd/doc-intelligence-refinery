from enum import Enum
from typing import List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class OriginType(str, Enum):
    """Represents the source nature of document content."""
    NATIVE_DIGITAL = "native_digital"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"


class LayoutComplexity(str, Enum):
    """Represents structural complexity of page layout."""
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"


class EstimatedCost(str, Enum):
    """Represents estimated processing cost tier for extraction."""
    FAST_TEXT_SUFFICIENT = "fast_text_sufficient"
    NEEDS_LAYOUT_MODEL = "needs_layout_model"
    NEEDS_VISION_MODEL = "needs_vision_model"


# --- NEW RUBRIC REQUIRED MODELS ---

class PageIndex(BaseModel):
    """Maps physical pages to logical content locations."""
    model_config = ConfigDict(extra="forbid")
    
    page_number: int = Field(ge=1)
    char_start: int = Field(description="Start character index in full text stream")
    char_end: int = Field(description="End character index in full text stream")


class ProvenanceChain(BaseModel):
    """Tracks the 'who, what, where' of an extracted piece of data."""
    model_config = ConfigDict(extra="forbid")
    
    strategy_used: str
    model_version: str = "v1.0"
    confidence_score: float = Field(ge=0.0, le=1.0)
    source_file: str
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class LDU(BaseModel):
    """Logical Document Unit (e.g., a Section, a Chapter, or a Table)."""
    model_config = ConfigDict(extra="forbid")
    
    unit_id: str
    unit_type: str  # e.g., "table", "paragraph", "header"
    content: str
    page_range: List[int]
    provenance: ProvenanceChain


# --- EXISTING COMPONENTS ---

class DocumentProfile(BaseModel):
    """Document-level profile used for strategy routing."""
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(description="Source PDF filename used for artifact naming and traceability.", min_length=1)
    origin_type: OriginType = Field(description="Origin classification.")
    layout_complexity: LayoutComplexity = Field(description="Structural complexity.")
    language: str = Field(description="Primary language.", min_length=1)
    domain_hint: str = Field(description="Business/domain hint.", min_length=1)
    estimated_cost: EstimatedCost = Field(description="Cost tier.")


class Table(BaseModel):
    """Structured representation of a table extracted from a page."""
    model_config = ConfigDict(extra="forbid")

    headers: List[str]
    rows: List[List[str]]
    title: Optional[str] = None


class ExtractedPage(BaseModel):
    """Page-level extraction payload."""
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str
    tables: List[Table] = Field(default_factory=list)
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class ExtractedDocument(BaseModel):
    """Master document contract for downstream indexing and analytics."""
    model_config = ConfigDict(extra="forbid")

    filename: str
    doc_id: str
    profile: DocumentProfile
    pages: List[ExtractedPage] = Field(default_factory=list)
    
    # New Field for Rubric
    logical_units: List[LDU] = Field(
        default_factory=list, 
        description="Collection of LDUs (Logical Document Units) like sections or tables."
    )
    
    strategy_used: str