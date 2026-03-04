import pdfplumber
from pathlib import Path

from src.models.document_schema import (
	DocumentProfile,
	EstimatedCost,
	LayoutComplexity,
	OriginType,
)


class TriageAgent:
	"""Profiles documents and routes them toward an extraction strategy."""

	DOMAIN_KEYWORDS = {
		"financial": [
			"inflation",
			"cbe",
			"birr",
			"balance sheet",
			"bank",
		],
		"legal": [
			"law",
			"regulation",
			"compliance",
			"statute",
		],
		"technical": [
			"system",
			"architecture",
			"api",
			"integration",
		],
		"medical": [
			"patient",
			"diagnosis",
			"treatment",
			"clinical",
		],
		"technical_legal": [
			"vulnerability",
			"standard",
			"procedure",
			"clause",
		],
	}

	def profile_document(self, pdf_path: str) -> DocumentProfile:
		"""Analyze the first page and return a DocumentProfile.

		Args:
			pdf_path: Path to the PDF document.

		Returns:
			A populated DocumentProfile based on first-page heuristics.
		"""
		filename = Path(pdf_path).name

		with pdfplumber.open(pdf_path) as pdf:
			if not pdf.pages:
				origin_type = OriginType.SCANNED_IMAGE
				layout_complexity = LayoutComplexity.MULTI_COLUMN
				domain_hint = "general"
				estimated_cost = EstimatedCost.NEEDS_VISION_MODEL
				return DocumentProfile(
					filename=filename,
					origin_type=origin_type,
					layout_complexity=layout_complexity,
					language="en",
					domain_hint=domain_hint,
					estimated_cost=estimated_cost,
				)

			page = pdf.pages[0]
			text = page.extract_text() or ""

			char_count = len(text)
			has_tables = bool(page.find_tables())
			image_count = len(page.images or [])
			page_area = float(page.width) * float(page.height)
			chars = page.objects.get("char", [])

			origin_type = self._detect_origin(char_count=char_count, image_count=image_count)
			layout_complexity = self._detect_layout_complexity(
				has_tables=has_tables,
				char_count=char_count,
				chars=chars,
				page_height=float(page.height),
			)
			domain_hint = self._detect_domain_hint(text)
			estimated_cost = self._estimate_cost(origin_type, layout_complexity)

			_ = page_area

			return DocumentProfile(
				filename=filename,
				origin_type=origin_type,
				layout_complexity=layout_complexity,
				language="en",
				domain_hint=domain_hint,
				estimated_cost=estimated_cost,
			)

	def _detect_origin(self, char_count: int, image_count: int) -> OriginType:
		if char_count < 50 and image_count > 0:
			return OriginType.SCANNED_IMAGE
		return OriginType.NATIVE_DIGITAL

	def _detect_layout_complexity(
		self,
		has_tables: bool,
		char_count: int,
		chars: list,
		page_height: float,
	) -> LayoutComplexity:
		if has_tables:
			return LayoutComplexity.TABLE_HEAVY

		if self._has_vertical_gutter(chars=chars, page_height=page_height):
			return LayoutComplexity.MULTI_COLUMN

		if char_count > 1500:
			return LayoutComplexity.SINGLE_COLUMN

		return LayoutComplexity.MULTI_COLUMN

	def _has_vertical_gutter(self, chars: list, page_height: float) -> bool:
		"""Detect a likely multi-column gutter between x=200 and x=400.

		A gutter is considered present when:
		- There is text on both the left and right sides of the gutter region.
		- The gutter region remains empty across most vertical slices of the page.
		"""
		if not chars or page_height <= 0:
			return False

		gutter_x_min, gutter_x_max = 200.0, 400.0
		left_char_count = 0
		right_char_count = 0
		gutter_vertical_spans: list[tuple[float, float]] = []

		for char in chars:
			x0 = float(char.get("x0", 0.0) or 0.0)
			x1 = float(char.get("x1", 0.0) or 0.0)
			top = float(char.get("top", 0.0) or 0.0)
			bottom = float(char.get("bottom", top) or top)

			if x1 < gutter_x_min:
				left_char_count += 1
			if x0 > gutter_x_max:
				right_char_count += 1

			intersects_gutter = not (x1 <= gutter_x_min or x0 >= gutter_x_max)
			if intersects_gutter:
				start = max(0.0, min(top, page_height))
				end = max(0.0, min(bottom, page_height))
				if end > start:
					gutter_vertical_spans.append((start, end))

		if left_char_count < 80 or right_char_count < 80:
			return False

		slice_count = 20
		slice_height = page_height / slice_count
		empty_slices = 0

		for idx in range(slice_count):
			slice_start = idx * slice_height
			slice_end = slice_start + slice_height
			has_gutter_char = any(
				not (span_end <= slice_start or span_start >= slice_end)
				for span_start, span_end in gutter_vertical_spans
			)
			if not has_gutter_char:
				empty_slices += 1

		empty_ratio = empty_slices / slice_count
		return empty_ratio >= 0.8

	def _detect_domain_hint(self, text: str) -> str:
		text_lower = text.lower()
		scores = {
			domain: 0
			for domain in self.DOMAIN_KEYWORDS
		}

		for domain, keywords in self.DOMAIN_KEYWORDS.items():
			for keyword in keywords:
				scores[domain] += text_lower.count(keyword)

		best_domain = max(scores, key=scores.get)
		if scores[best_domain] == 0:
			return "general"

		return best_domain

	def _estimate_cost(
		self,
		origin_type: OriginType,
		layout_complexity: LayoutComplexity,
	) -> EstimatedCost:
		if origin_type == OriginType.SCANNED_IMAGE:
			return EstimatedCost.NEEDS_VISION_MODEL
		if layout_complexity == LayoutComplexity.TABLE_HEAVY:
			return EstimatedCost.NEEDS_LAYOUT_MODEL
		return EstimatedCost.FAST_TEXT_SUFFICIENT
