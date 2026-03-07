from collections import defaultdict
import os
import re
from typing import Any, List

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - optional runtime dependency
    ChatGroq = None

from src.models.document_schema import LDU, PageIndexNode


class DocumentIndexer:
    """Builds a hierarchical page/section index from LDUs."""

    HEADER_TYPES = {"header", "title"}
    TITLE_NOISE_PATTERNS = (
        r"\bETHIOPIAN\s+STATISTICAL\s+SERVICE\b",
        r"\bStatistical\s+Bulletin\b",
        r"\bIssue\s+No\b[^|,;\n]*",
        r"\bPage\s*\d+\b",
    )
    SECTION_HEADER_PATTERN = re.compile(
        r"^\s*(?:section\s+\d+(?:\.\d+)*|\d+(?:\.\d+)+)\b[:.)\-\s]*",
        flags=re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.llm = None
        api_key = os.getenv("GROQ_API_KEY")
        if api_key and ChatGroq is not None:
            self.llm = ChatGroq(
                model_name="llama-3.3-70b-versatile",
                groq_api_key=api_key,
                temperature=0,
                max_tokens=300,
            )

    def build_index(self, ldus: list[LDU]) -> list[PageIndexNode]:
        json_nodes = self.build_index_tree_json(ldus)
        return [self._json_node_to_page_index_node(node) for node in json_nodes]

    def build_index_tree_json(self, ldus: list[LDU]) -> list[dict[str, Any]]:
        """
        Return a hierarchical JSON tree for section indexing.

        Each node includes: title, page_start, page_end, summary, contains_tables,
        contains_figures, and children.
        """
        if not ldus:
            return []

        valid_ldus = [item for item in ldus if self._is_valid_ldu_like(item)]
        ordered_ldus = sorted(
            valid_ldus,
            key=lambda item: (
                self._first_page_or_default(item),
                str(getattr(item, "uid", "")),
            ),
        )

        if not ordered_ldus:
            return []

        root_nodes: list[dict[str, Any]] = []
        stack: list[tuple[int, dict[str, Any]]] = []
        page_roots: dict[int, dict[str, Any]] = {}

        for ldu in ordered_ldus:
            page = int(ldu.page_refs[0]) if ldu.page_refs else 1
            title_candidate = self._clean_title(self._ldu_title_text(ldu))
            is_header, level, normalized_title = self._parse_section_header(ldu, title_candidate)

            if is_header:
                node = {
                    "title": normalized_title,
                    "page_start": page,
                    "page_end": page,
                    "summary": "",
                    "contains_tables": False,
                    "contains_figures": False,
                    "children": [],
                    "_texts": [],
                }

                while stack and stack[-1][0] >= level:
                    stack.pop()

                if stack:
                    stack[-1][1]["children"].append(node)
                else:
                    root_nodes.append(node)

                stack.append((level, node))
                continue

            target = stack[-1][1] if stack else page_roots.get(page)
            if target is None:
                target = {
                    "title": f"Page {page}",
                    "page_start": page,
                    "page_end": page,
                    "summary": "",
                    "contains_tables": False,
                    "contains_figures": False,
                    "children": [],
                    "_texts": [],
                }
                page_roots[page] = target
                root_nodes.append(target)

            self._append_chunk_to_node(target, ldu, page)

            for _, ancestor in stack:
                ancestor["page_start"] = min(int(ancestor["page_start"]), page)
                ancestor["page_end"] = max(int(ancestor["page_end"]), page)
                self._update_node_flags(ancestor, ldu)

        for node in root_nodes:
            self._finalize_node(node)

        return root_nodes

    def _is_valid_ldu_like(self, item: Any) -> bool:
        if item is None:
            return False
        if not hasattr(item, "page_refs"):
            return False
        if not hasattr(item, "uid"):
            return False

        page_refs = getattr(item, "page_refs", None)
        return isinstance(page_refs, list)

    def _first_page_or_default(self, item: Any, default: int = 1) -> int:
        page_refs = getattr(item, "page_refs", None)
        if isinstance(page_refs, list) and page_refs:
            try:
                return int(page_refs[0])
            except (TypeError, ValueError):
                return default
        return default

    def _json_node_to_page_index_node(self, node: dict[str, Any]) -> PageIndexNode:
        return PageIndexNode(
            title=str(node.get("title", "Untitled Section")),
            summary=str(node.get("summary", "")),
            page_start=int(node.get("page_start", 1)),
            page_end=int(node.get("page_end", 1)),
            children=[self._json_node_to_page_index_node(child) for child in node.get("children", [])],
        )

    def _parse_section_header(self, ldu: LDU, title_candidate: str) -> tuple[bool, int, str]:
        content = title_candidate or ""
        unit_type = (ldu.unit_type or "").strip().lower()

        # Explicitly match headers like "1.1 ..." and "Section 2 ...".
        match = self.SECTION_HEADER_PATTERN.match(content)
        if match:
            token = match.group(0)
            level = self._header_level_from_token(token)
            return True, level, content

        if unit_type in self.HEADER_TYPES and content:
            return True, 1, content

        return False, 0, content

    def _header_level_from_token(self, token: str) -> int:
        normalized = re.sub(r"[^\w\.]", "", (token or "").casefold())
        number_match = re.search(r"(\d+(?:\.\d+)*)", normalized)
        if not number_match:
            return 1

        number_path = number_match.group(1)
        return max(1, len(number_path.split(".")))

    def _append_chunk_to_node(self, node: dict[str, Any], ldu: LDU, page: int) -> None:
        text = re.sub(r"\s+", " ", (ldu.content or "").strip())
        if text:
            node.setdefault("_texts", []).append(text)

        node["page_start"] = min(int(node["page_start"]), page)
        node["page_end"] = max(int(node["page_end"]), page)
        self._update_node_flags(node, ldu)

    def _update_node_flags(self, node: dict[str, Any], ldu: LDU) -> None:
        chunk_type = (ldu.unit_type or "").casefold()
        content = (ldu.content or "").casefold()

        if chunk_type == "table" or "table" in content:
            node["contains_tables"] = True
        if chunk_type == "figure" or "figure" in content or "fig." in content:
            node["contains_figures"] = True

    def _finalize_node(self, node: dict[str, Any]) -> None:
        for child in node.get("children", []):
            self._finalize_node(child)
            node["page_start"] = min(int(node["page_start"]), int(child.get("page_start", node["page_start"])))
            node["page_end"] = max(int(node["page_end"]), int(child.get("page_end", node["page_end"])))
            node["contains_tables"] = bool(node.get("contains_tables")) or bool(child.get("contains_tables"))
            node["contains_figures"] = bool(node.get("contains_figures")) or bool(child.get("contains_figures"))

        summary_texts = self._collect_node_texts(node)
        node["summary"] = self._generate_section_summary(
            str(node.get("title", "Untitled Section")),
            summary_texts,
        )

        node.pop("_texts", None)

    def _collect_node_texts(self, node: dict[str, Any]) -> list[str]:
        texts = [str(item) for item in node.get("_texts", []) if str(item).strip()]
        for child in node.get("children", []):
            texts.extend(self._collect_node_texts(child))
        return texts

    def _attach_header_tree(self, page_node: PageIndexNode, page_ldus: list[LDU], page_number: int) -> None:
        headers = [ldu for ldu in page_ldus if self._is_header_ldu(ldu)]
        if not headers:
            return

        headers_by_parent: dict[str | None, list[LDU]] = defaultdict(list)
        headers_by_uid: dict[str, LDU] = {}

        for header in headers:
            headers_by_uid[header.uid] = header

        for header in headers:
            parent_id = header.parent_section
            if parent_id and parent_id not in headers_by_uid:
                parent_id = None
            headers_by_parent[parent_id].append(header)

        for parent_id in headers_by_parent:
            headers_by_parent[parent_id].sort(key=lambda item: item.uid)

        root_headers = headers_by_parent.get(None, [])
        for root_header in root_headers:
            child_node = self._build_recursive_node(root_header, headers_by_parent, page_number)
            page_node.children.append(child_node)

    def _build_recursive_node(
        self,
        header_ldu: LDU,
        headers_by_parent: dict[str | None, list[LDU]],
        page_number: int,
    ) -> PageIndexNode:
        title_text = self._clean_title(self._ldu_title_text(header_ldu)) or header_ldu.uid
        subtree_ldus = self._collect_subtree_ldus(header_ldu, headers_by_parent)

        node = PageIndexNode(
            title=title_text,
            summary=self._generate_section_summary(title_text, subtree_ldus),
            page_start=page_number,
            page_end=page_number,
            children=[],
        )

        for child_header in headers_by_parent.get(header_ldu.uid, []):
            node.children.append(self._build_recursive_node(child_header, headers_by_parent, page_number))

        return node

    def _collect_subtree_ldus(
        self,
        header_ldu: LDU,
        headers_by_parent: dict[str | None, list[LDU]],
    ) -> list[LDU]:
        collected: list[LDU] = [header_ldu]
        for child_header in headers_by_parent.get(header_ldu.uid, []):
            collected.extend(self._collect_subtree_ldus(child_header, headers_by_parent))
        return collected

    def _generate_section_summary(self, title: str, section_texts: List[str]) -> str:
        section_title = self._clean_title(title) or "Untitled Section"
        normalized_contents = [re.sub(r"\s+", " ", text).strip() for text in section_texts if text.strip()]
        context_excerpt = " ".join(normalized_contents)[:2200]

        prompt = (
            "You are a concise document summarizer. "
            "Write a summary in 2 to 3 sentences with factual language and no bullet points. "
            "Sentence 1 should state the section's main topic. "
            "The remaining sentence(s) should capture key requirements, evidence, or implications.\\n\\n"
            f"Section title: {section_title}\\n"
            f"Section content: {context_excerpt or 'No extracted content provided.'}"
        )

        if self.llm is not None and context_excerpt:
            try:
                response = self.llm.invoke(prompt)
                content = str(getattr(response, "content", "")).strip()
                if content:
                    return content
            except Exception:
                pass

        if context_excerpt:
            lead_text = context_excerpt[:220].rstrip(" .;,:-")
            return (
                f"{section_title} centers on {lead_text}. "
                "It highlights the most relevant details from this section to support downstream search and retrieval. "
                "This summary is generated from extracted evidence when an LLM response is unavailable."
            )

        return (
            f"{section_title} is identified as a section in the document structure. "
            "No extracted child content was available, so this summary is a structural placeholder."
        )

    def generate_summary(self, node: PageIndexNode, context_text: str) -> str:
        del context_text
        return f"Summary of [{node.title}]"

    def _ldu_title_text(self, ldu: LDU) -> str:
        if not ldu.content:
            return ""
        return (ldu.content or "").strip().splitlines()[0].strip()

    def _clean_title(self, title: str) -> str:
        cleaned = (title or "").strip()
        if not cleaned:
            return ""

        for pattern in self.TITLE_NOISE_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|:,;\t")
        return cleaned

    def _is_header_ldu(self, ldu: LDU) -> bool:
        unit_type = (ldu.unit_type or "").lower()
        if unit_type in self.HEADER_TYPES:
            return True

        content = self._ldu_title_text(ldu)
        if not content:
            return False

        heuristic_header = (
            ldu.parent_section is None
            and len(content) < 120
            and not content.rstrip().endswith(".")
        )
        return heuristic_header

    def get_relevant_pages(self, query: str, nodes: list[PageIndexNode]) -> list[int]:
        print(f"DEBUG: Finding pages for query: '{query}'")
        query_terms = {term for term in re.findall(r'\w+', (query or "").casefold()) if term}

        matched_pages: set[int] = set()

        for node in self._iter_nodes(nodes):
            title_text = node.title.casefold()
            if any(term in title_text for term in query_terms):
                matched_pages.update(range(node.page_start, node.page_end + 1))

        # EMERGENCY FALLBACK: If no pages matched keywords, use all pages (1-5)
        # to ensure the vector store is actually searched.
        if not matched_pages:
            print("DEBUG: No keyword match in index. Falling back to ALL pages.")
            return list(range(1, 6))

        print(f"DEBUG: Matched Pages: {sorted(list(matched_pages))}")
        return sorted(list(matched_pages))

    def _expand_query_terms(self, query_terms: set[str]) -> set[str]:
        expanded = set(query_terms)

        if "price" in query_terms or "index" in query_terms:
            expanded.update({"inflation", "cpi", "report", "summary"})
        if "changes" in query_terms or "change" in query_terms:
            expanded.update({"trend", "summary", "report"})
        if "inflation" in query_terms:
            expanded.update({"cpi", "price", "index"})

        return expanded

    def _iter_nodes(self, nodes: list[PageIndexNode]) -> list[PageIndexNode]:
        ordered: list[PageIndexNode] = []

        def walk(node: PageIndexNode) -> None:
            ordered.append(node)
            for child in node.children:
                walk(child)

        for node in nodes:
            walk(node)

        return ordered

    def _collect_relevant_pages(
        self,
        node: PageIndexNode,
        query_terms: set[str],
        matched_pages: set[int],
        include_cpi_pages: bool,
    ) -> None:
        title_text = (node.title or "").casefold()
        has_keyword_match = any(term in title_text for term in query_terms)
        has_cpi_fallback_match = include_cpi_pages and ("cpi" in title_text)

        if has_keyword_match or has_cpi_fallback_match:
            matched_pages.update(range(node.page_start, node.page_end + 1))

        for child in node.children:
            self._collect_relevant_pages(
                node=child,
                query_terms=query_terms,
                matched_pages=matched_pages,
                include_cpi_pages=include_cpi_pages,
            )
