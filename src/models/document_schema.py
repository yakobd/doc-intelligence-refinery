from enum import Enum
from typing import List, Optional

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


class DocumentProfile(BaseModel):
	"""Document-level profile used for strategy routing."""

	model_config = ConfigDict(extra="forbid")

	origin_type: OriginType = Field(
		description="Origin classification of the document content: native digital, scanned image, or mixed.",
	)
	layout_complexity: LayoutComplexity = Field(
		description="Layout complexity hint used for extraction strategy selection.",
	)
	language: str = Field(
		description="Primary language of the document text (for example: en, am, or multilingual).",
		min_length=1,
	)
	domain_hint: str = Field(
		description="Business/domain hint such as financial, legal, policy, or operations.",
		min_length=1,
	)
	estimated_cost: EstimatedCost = Field(
		description="Estimated extraction cost tier indicating whether text, layout, or vision-heavy processing is required.",
	)


class Table(BaseModel):
	"""Structured representation of a table extracted from a page."""

	model_config = ConfigDict(extra="forbid")

	headers: List[str] = Field(
		description="Ordered list of table column headers.",
	)
	rows: List[List[str]] = Field(
		description="Table body rows where each inner list represents one row of cell values.",
	)
	title: Optional[str] = Field(
		default=None,
		description="Optional table title or caption when available in the source document.",
	)


class ExtractedPage(BaseModel):
	"""Page-level extraction payload containing text and structured artifacts."""

	model_config = ConfigDict(extra="forbid")

	page_number: int = Field(
		description="1-based page number in the source document.",
		ge=1,
	)
	text: str = Field(
		description="Extracted plain text content for this page.",
	)
	tables: List[Table] = Field(
		default_factory=list,
		description="Structured tables extracted from this page.",
	)
	extraction_confidence: float = Field(
		description="Confidence score for page extraction quality, constrained to the range [0, 1].",
		ge=0.0,
		le=1.0,
	)


class ExtractedDocument(BaseModel):
	"""Master document contract for downstream indexing and analytics."""

	model_config = ConfigDict(extra="forbid")

	filename: str = Field(
		description="Original filename of the processed document.",
		min_length=1,
	)
	doc_id: str = Field(
		description="Unique identifier assigned to the document in the refinery pipeline.",
		min_length=1,
	)
	profile: DocumentProfile = Field(
		description="Document-level profile containing routing and cost signals.",
	)
	pages: List[ExtractedPage] = Field(
		default_factory=list,
		description="Ordered list of extracted pages and their structured content.",
	)
	strategy_used: str = Field(
		description="Name of the extraction strategy used for the document (for example: Strategy A, Strategy B, or Strategy C).",
		min_length=1,
	)
