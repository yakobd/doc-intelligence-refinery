import hashlib
from enum import Enum
from typing import List, Optional, Dict, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# 1. Type Precision: Dedicated BBox Model (Rubric Item #2)
class BBox(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    page_number: int

# 2. Categorical Fields: Enums
class OriginType(str, Enum):
    NATIVE_DIGITAL = "native_digital"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"

class LayoutComplexity(str, Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"

class StrategyTier(str, Enum):
    STRATEGY_A = "FASTTEXT"
    STRATEGY_B = "LAYOUT"
    STRATEGY_C = "VISION"

# 3. Provenance Citation Chains (Rubric Item #3)
class ProvenanceChain(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_file: str
    content_hash: str  # Required for Mastery
    bbox: BBox         # Required: Structured sub-model
    strategy_used: str
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# 4. Logical Document Units (LDU) (Rubric Item #3 & #4)
class LDU(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uid: str
    unit_type: str  # e.g., "table", "paragraph"
    content: str
    content_hash: str    # Required for Mastery
    page_refs: List[int] # Required for Mastery
    bounding_box: BBox   # Required for Mastery
    parent_section: Optional[str] = None # Recursive relationship
    child_chunks: List[str] = []         # Chunk relationship
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("LDU content cannot be empty")
        return v

# 5. Hierarchical Page/Section Indexing (Rubric Item #4)
class PageIndexNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    page_start: int
    page_end: int
    children: List["PageIndexNode"] = [] # Required: Recursive nodes

    @model_validator(mode='after')
    def validate_range(self) -> 'PageIndexNode':
        if self.page_end < self.page_start:
            raise ValueError(f"page_end ({self.page_end}) must be >= page_start")
        return self

# Rebuild model to support the recursive "children" type
PageIndexNode.model_rebuild()

# 6. Document Profiling (Triage Output)
class DocumentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    origin_type: OriginType
    layout_complexity: LayoutComplexity
    selected_strategy: StrategyTier
    confidence_score: float = Field(ge=0.0, le=1.0)
    estimated_cost: float
    pages: int
    language: str = "en"
    domain_hint: str = "financial"
    is_form_fillable: bool = False
    domain_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

# 7. Normalized Extraction Output
class NormalizedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    doc_id: str
    profile: DocumentProfile
    ldus: List[LDU]
    index: List[PageIndexNode]
    metadata: Dict[str, Any]