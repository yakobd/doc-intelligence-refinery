import base64
import hashlib
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from src.models.document_schema import LDU
except ImportError:
    LDU = None

try:
    from src.agents.triage import TriageAgent
except ImportError:
    TriageAgent = None

try:
    # QueryAgent naming requested by UI spec maps to QueryOrchestrator in this codebase.
    from src.agents.query_agent import QueryOrchestrator as QueryAgent
except ImportError:
    QueryAgent = None

try:
    from src.engines.indexer import DocumentIndexer
except ImportError:
    DocumentIndexer = None


PIPELINE_STAGES = [
    "Triage",
    "Extraction",
    "Chunking",
    "Indexing",
    "Query Ready",
]


@dataclass
class ProvenanceReceipt:
    source_file_name: str
    page_number: int
    content_hash: str
    bbox: list[float]


class RefineryBackend:
    """Backend adapter that uses real triage/query agents and mocks the remaining pipeline UI data."""

    def __init__(self) -> None:
        self.triage_agent_cls = TriageAgent
        self.query_agent_cls = QueryAgent

    def run_pipeline(self, pdf_path: Path, pdf_name: str, pdf_bytes: bytes) -> dict[str, Any]:
        triage = self._run_triage(pdf_path, pdf_name, pdf_bytes)
        ledger = self._mock_extraction_ledger(pdf_name, triage["profile"], triage["strategy"], pdf_path)
        page_index = self._mock_page_index(ledger)
        # Rebuild query agent when a new pipeline result is generated.
        st.session_state.query_agent = self._build_query_agent(page_index)
        return {
            "triage": triage,
            "ledger": ledger,
            "page_index": page_index,
        }

    def answer_question(self, question: str, result_bundle: dict[str, Any]) -> tuple[str, ProvenanceReceipt]:
        agent = st.session_state.get("query_agent")
        if agent is None:
            agent = self._build_query_agent(result_bundle.get("page_index", []))
            st.session_state.query_agent = agent
        if agent is not None:
            try:
                state = agent.run(question)
                answer = str(state.get("final_response", "No answer generated."))
                receipt = self._receipt_from_state(state, result_bundle)
                return answer, receipt
            except Exception:
                pass

        # Fallback if QueryAgent cannot initialize/run in the current environment.
        ledger = result_bundle.get("ledger", [])
        selected = ledger[0] if ledger else {}
        fallback_answer = (
            "QueryAgent could not be executed in this runtime, so this is a safe fallback response.\n\n"
            f"Top extracted span:\n{str(selected.get('content', 'No extracted content available yet.'))[:240]}"
        )
        fallback_receipt = ProvenanceReceipt(
            source_file_name=str(selected.get("source_file", "unknown.pdf")),
            page_number=int(selected.get("page_refs", [1])[0]),
            content_hash=str(selected.get("content_hash", "")),
            bbox=[float(v) for v in selected.get("bounding_box", [0, 0, 0, 0])],
        )
        return fallback_answer, fallback_receipt

    def _run_triage(self, pdf_path: Path, pdf_name: str, pdf_bytes: bytes) -> dict[str, str]:
        triage_agent = st.session_state.get("triage_agent")
        if triage_agent is not None:
            try:
                profile = triage_agent.profile_document(str(pdf_path))
                strategy = self._to_strategy_label(str(profile.selected_strategy.value))
                doc_class = self._to_document_class(str(profile.origin_type.value), str(profile.layout_complexity.value))
                return {"profile": doc_class, "strategy": strategy}
            except Exception:
                pass

        # Fallback classification if triage is unavailable.
        digest = hashlib.md5(pdf_bytes).hexdigest()
        classes = ["Class A", "Class B", "Class C", "Class D"]
        strategies = ["Strategy A", "Strategy B", "Strategy C"]

        profile = classes[int(digest[0], 16) % len(classes)]
        strategy = strategies[int(digest[1], 16) % len(strategies)]
        return {"profile": profile, "strategy": strategy}

    def _build_query_agent(self, page_index_json: list[dict[str, Any]]) -> Any | None:
        if self.query_agent_cls is None:
            return None

        try:
            # QueryAgent (QueryOrchestrator) accepts plain index nodes and builds internal tools.
            return self.query_agent_cls(page_index=page_index_json)
        except Exception:
            return None

    def _receipt_from_state(self, state: dict[str, Any], result_bundle: dict[str, Any]) -> ProvenanceReceipt:
        provenance = state.get("final_provenance")
        if provenance is not None:
            source_file_name = str(getattr(provenance, "document_name", "unknown.pdf"))
            page_number = int(getattr(provenance, "page_number", 1))
            content_hash = str(getattr(provenance, "content_hash", ""))
            bbox = [float(v) for v in getattr(provenance, "bbox", [0, 0, 0, 0])]
            return ProvenanceReceipt(
                source_file_name=source_file_name,
                page_number=page_number,
                content_hash=content_hash,
                bbox=bbox,
            )

        ledger = result_bundle.get("ledger", [])
        selected = ledger[0] if ledger else {}
        return ProvenanceReceipt(
            source_file_name=str(selected.get("source_file", "unknown.pdf")),
            page_number=int(selected.get("page_refs", [1])[0]),
            content_hash=str(selected.get("content_hash", "")),
            bbox=[float(v) for v in selected.get("bounding_box", [0, 0, 0, 0])],
        )

    def _to_strategy_label(self, strategy_value: str) -> str:
        mapping = {
            "FASTTEXT": "Strategy A",
            "LAYOUT": "Strategy B",
            "VISION": "Strategy C",
        }
        return mapping.get(strategy_value.upper(), "Strategy A")

    def _to_document_class(self, origin_type: str, layout_complexity: str) -> str:
        origin = origin_type.casefold()
        layout = layout_complexity.casefold()

        if origin == "native_digital" and layout == "single_column":
            return "Class A"
        if origin == "native_digital" and layout in {"multi_column", "table_heavy"}:
            return "Class B"
        if origin == "mixed":
            return "Class C"
        return "Class D"

    def _mock_extraction_ledger(
        self,
        pdf_name: str,
        profile: str,
        strategy: str,
        pdf_path: Path,
    ) -> list[dict[str, Any]]:
        page_count = self._count_pages(pdf_path)
        ledger: list[dict[str, Any]] = []
        for page in range(1, min(page_count, 6) + 1):
            section_title = f"Section {page}.0 - Refinery Findings"
            content = (
                f"{section_title}. This is a mocked LDU extracted from {pdf_name}. "
                f"Profile={profile}, Strategy={strategy}, Page={page}."
            )
            content_hash = hashlib.sha1(content.encode("utf-8")).hexdigest()
            ledger.append(
                {
                    "uid": f"ldu-{page:03d}",
                    "chunk_type": "paragraph" if page % 2 else "table",
                    "content": content,
                    "content_hash": content_hash,
                    "page_refs": [page],
                    "bounding_box": [0.08, 0.1, 0.92, 0.28],
                    "source_file": pdf_name,
                }
            )
        return ledger

    def _mock_page_index(self, ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if DocumentIndexer is None or LDU is None:
            return [
                {
                    "title": "Executive Overview",
                    "summary": "Mocked summary generated without indexer dependency.",
                    "page_start": 1,
                    "page_end": max(1, len(ledger)),
                    "children": [],
                }
            ]

        try:
            ldu_models = [LDU(**row) for row in ledger]
            indexer = DocumentIndexer()
            return indexer.build_index_tree_json(ldu_models)
        except Exception:
            return [
                {
                    "title": "Index Build Fallback",
                    "summary": "Page index could not be generated by DocumentIndexer in mock mode.",
                    "page_start": 1,
                    "page_end": max(1, len(ledger)),
                    "children": [],
                }
            ]

    def _count_pages(self, pdf_path: Path) -> int:
        if pdfplumber is None:
            return 3
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                return max(1, len(pdf.pages))
        except Exception:
            return 3


def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #0f1724;
                --panel: #152136;
                --panel-2: #1c2c47;
                --text: #e8eef9;
                --muted: #9db1cc;
                --accent: #00b8d4;
                --ok: #22c55e;
            }
            .stApp {
                background: radial-gradient(circle at 20% 15%, #1f2e4a 0%, #0f1724 55%);
                color: var(--text);
            }
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #101a2d 0%, #0a1222 100%);
            }
            .triage-card {
                background: linear-gradient(135deg, #1d3052 0%, #152136 100%);
                border: 1px solid #314b72;
                border-radius: 12px;
                padding: 0.8rem 1rem;
                margin-bottom: 0.8rem;
            }
            .ledger-scroll {
                max-height: 560px;
                overflow-y: auto;
                border: 1px solid #2a3e62;
                border-radius: 10px;
                padding: 0.5rem;
                background: rgba(16, 26, 45, 0.9);
            }
            .ledger-item {
                border: 1px solid #29405f;
                border-radius: 8px;
                margin-bottom: 0.6rem;
                padding: 0.5rem;
                background: rgba(28, 44, 71, 0.75);
            }
            .receipt {
                border: 1px solid #2c4e67;
                border-radius: 10px;
                padding: 0.75rem;
                background: rgba(18, 34, 52, 0.9);
            }
            iframe {
                border-radius: 10px;
                border: 1px solid #274368;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("pipeline_result", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("uploaded_name", None)
    st.session_state.setdefault("pdf_bytes", b"")
    st.session_state.setdefault("triage_agent", None)
    st.session_state.setdefault("query_agent", None)


def init_agents_once(backend: RefineryBackend) -> None:
    if st.session_state.triage_agent is None and backend.triage_agent_cls is not None:
        try:
            st.session_state.triage_agent = backend.triage_agent_cls()
        except Exception:
            st.session_state.triage_agent = None


def render_pdf(pdf_bytes: bytes) -> None:
    data = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_viewer = f"""
        <iframe src="data:application/pdf;base64,{data}" width="100%" height="780"></iframe>
    """
    st.components.v1.html(pdf_viewer, height=800, scrolling=True)


def render_ledger(ledger: list[dict[str, Any]]) -> None:
    st.markdown('<div class="ledger-scroll">', unsafe_allow_html=True)
    for index, ldu in enumerate(ledger, start=1):
        st.markdown(f'<div class="ledger-item"><strong>LDU {index}</strong></div>', unsafe_allow_html=True)
        st.json(ldu, expanded=False)
    st.markdown("</div>", unsafe_allow_html=True)


def run_pipeline_with_status(backend: RefineryBackend, temp_pdf_path: Path, pdf_name: str, pdf_bytes: bytes) -> dict[str, Any]:
    progress_bar = st.progress(0, text="Pipeline idle")
    with st.status("Running Doc-Intelligence Refinery", expanded=True) as status:
        for i, stage in enumerate(PIPELINE_STAGES, start=1):
            status.write(f"{stage} stage in progress...")
            progress_bar.progress(i / len(PIPELINE_STAGES), text=f"{stage} ({i}/{len(PIPELINE_STAGES)})")
            time.sleep(0.25)

        result = backend.run_pipeline(temp_pdf_path, pdf_name, pdf_bytes)
        status.update(label="Pipeline completed: Query Ready", state="complete")

    return result


def main() -> None:
    st.set_page_config(page_title="Doc-Intelligence Refinery Dashboard", page_icon=":mag:", layout="wide")
    apply_custom_theme()
    init_state()

    st.title("Doc-Intelligence Refinery")
    st.caption("Agentic pipeline observability: Triage -> Extraction -> Chunking -> Indexing -> Query Ready")

    backend = RefineryBackend()
    init_agents_once(backend)

    with st.sidebar:
        st.header("Upload")
        uploaded_pdf = st.file_uploader("Choose a PDF", type=["pdf"])
        st.write("Upload a document to run the mocked refinery pipeline.")

        if uploaded_pdf and st.button("Run Pipeline", use_container_width=True):
            pdf_bytes = uploaded_pdf.read()
            st.session_state.uploaded_name = uploaded_pdf.name
            st.session_state.pdf_bytes = pdf_bytes
            st.session_state.chat_history = []
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_bytes)
                temp_path = Path(temp_file.name)

            st.session_state.pipeline_result = run_pipeline_with_status(
                backend=backend,
                temp_pdf_path=temp_path,
                pdf_name=uploaded_pdf.name,
                pdf_bytes=pdf_bytes,
            )

    result = st.session_state.pipeline_result

    if not result:
        st.info("Upload a PDF in the sidebar and click 'Run Pipeline' to begin.")
        return

    triage = result["triage"]
    ledger = result["ledger"]
    page_index = result["page_index"]

    st.markdown('<div class="triage-card">', unsafe_allow_html=True)
    metric_col_1, metric_col_2 = st.columns(2)
    metric_col_1.metric("Document Profile", triage["profile"])
    metric_col_2.metric("Selected Strategy", triage["strategy"])
    st.markdown("</div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.1, 1.0], gap="large")

    with left_col:
        st.subheader("PDF Viewer")
        if st.session_state.uploaded_name:
            st.write(f"File: {st.session_state.uploaded_name}")
        if st.session_state.pdf_bytes:
            render_pdf(st.session_state.pdf_bytes)
        else:
            st.warning("PDF preview not available for this rerun. Re-run pipeline after upload.")

    with right_col:
        st.subheader("Extraction Ledger (LDU JSON)")
        render_ledger(ledger)

    st.subheader("PageIndex Explorer")
    st.json(page_index, expanded=1)

    st.subheader("Ask the Document")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("receipt"):
                receipt = message["receipt"]
                st.markdown('<div class="receipt">', unsafe_allow_html=True)
                st.markdown("**Provenance Receipt**")
                st.write(f"Source File Name: `{receipt['source_file_name']}`")
                st.write(f"Page Number: `{receipt['page_number']}`")
                st.write(f"Content Hash: `{receipt['content_hash']}`")
                st.code(
                    json.dumps({"bbox": receipt["bbox"]}, indent=2),
                    language="json",
                )
                st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask a question about the uploaded document...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        answer, receipt = backend.answer_question(prompt, result)
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "receipt": {
                    "source_file_name": receipt.source_file_name,
                    "page_number": receipt.page_number,
                    "content_hash": receipt.content_hash,
                    "bbox": receipt.bbox,
                },
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()
