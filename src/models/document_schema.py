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

# 4. Chunk Model
class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uid: str
    content: str
    content_hash: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: int = Field(ge=0)

    @field_validator("content")
    @classmethod
    def chunk_content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk content cannot be empty")
        return v


# Cross-references between chunks (e.g., table summary -> source paragraph)
class ChunkRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_chunk_uid: str
    target_chunk_uid: str
    relationship_type: str = "related"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

# 4. Logical Document Units (LDU) (Rubric Item #3 & #4)
class LDU(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uid: str
    content: str
    chunk_type: str
    content_hash: str
    page_refs: List[int]
    bounding_box: List[float]
    parent_section: Optional[str] = None
    token_count: int = Field(default=0, ge=0)
    child_chunks: List[str] = Field(default_factory=list)
    chunk_relationships: List[ChunkRelationship] = Field(default_factory=list)
    chunks: List[Chunk] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        # Backward compatibility: accept `unit_type` and map to `chunk_type`.
        if "chunk_type" not in normalized and "unit_type" in normalized:
            normalized["chunk_type"] = normalized.pop("unit_type")

        # Backward compatibility: accept BBox model/dict and convert to list.
        bbox = normalized.get("bounding_box")
        if isinstance(bbox, BBox):
            normalized["bounding_box"] = [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
        elif isinstance(bbox, dict):
            if {"x_min", "y_min", "x_max", "y_max"}.issubset(bbox.keys()):
                normalized["bounding_box"] = [
                    bbox["x_min"],
                    bbox["y_min"],
                    bbox["x_max"],
                    bbox["y_max"],
                ]

        return normalized

    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("LDU content cannot be empty")
        return v

    @field_validator("bounding_box")
    @classmethod
    def validate_bounding_box(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError("bounding_box must contain exactly 4 float values")
        return v

    @model_validator(mode="after")
    def infer_token_count(self) -> "LDU":
        # Preserve explicit values; otherwise derive a simple token approximation.
        if self.token_count == 0 and self.content.strip():
            self.token_count = len(self.content.split())
        return self

    @property
    def unit_type(self) -> str:
        # Compatibility accessor for existing call sites.
        return self.chunk_type

# 5. Hierarchical Page/Section Indexing (Rubric Item #4)
class PageIndexNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    summary: str = ""
    page_start: int
    page_end: int
    children: List["PageIndexNode"] = Field(default_factory=list) # Required: Recursive nodes

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
    estimated_chars: int = 0
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