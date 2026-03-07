from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import argparse
from pathlib import Path
from typing import Any

from src.agents.chunker import SemanticChunker
from src.agents.fact_extractor import FactTableExtractor
from src.agents.indexer import DocumentIndexer
from src.agents.query_agent import QueryOrchestrator

import os
import ssl

# This tells the system to ignore SSL verification errors
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

try:
    # Requested import path in task prompt.
    from src.agents.triage import ExtractionRouter  # type: ignore[attr-defined]
except ImportError:
    # Fallback to the concrete implementation location.
    from src.agents.extractor import ExtractionRouter

from src.engines.vector_store import VectorStoreManager
from src.models.document_schema import Chunk, LDU, ProvenanceChain


def _to_raw_extractions(ldus: list[LDU], filename: str) -> list[dict[str, Any]]:
    raw_extractions: list[dict[str, Any]] = []
    for ldu in ldus:
        raw_extractions.append(
            {
                "uid": ldu.uid,
                "content": ldu.content,
                "chunk_type": ldu.chunk_type,
                "page": ldu.page_refs[0] if ldu.page_refs else 1,
                "bounding_box": ldu.bounding_box,
                "parent_section": ldu.parent_section,
                "token_count": ldu.token_count,
                "child_chunks": list(ldu.child_chunks),
                "filename": filename,
                "document_name": filename,
            }
        )
    return raw_extractions


def _ldus_to_chunks(ldus: list[LDU], filename: str) -> list[Chunk]:
    chunks: list[Chunk] = []

    for ldu in ldus:
        content = (ldu.content or "").strip()
        if not content:
            continue

        page_refs = [int(page) for page in ldu.page_refs]
        token_count = ldu.token_count if ldu.token_count > 0 else len(content.split())
        chunks.append(
            Chunk(
                uid=ldu.content_hash,
                content=content,
                content_hash=ldu.content_hash,
                metadata={
                    "filename": filename,
                    "parent_ldu_id": ldu.uid,
                    "unit_type": ldu.chunk_type,
                    "title": ldu.parent_section or (f"Page {page_refs[0]}" if page_refs else "Page 1"),
                    "page_numbers": page_refs,
                    "page_refs": page_refs,
                    "bbox": list(ldu.bounding_box),
                    "content_hash": ldu.content_hash,
                },
                token_count=token_count,
            )
        )

    return chunks


def _parse_query_and_hash(user_input: str) -> tuple[str, str | None]:
    text = (user_input or "").strip()
    if "|hash=" not in text:
        return text, None

    query_part, hash_part = text.split("|hash=", maxsplit=1)
    parsed_query = query_part.strip()
    parsed_hash = hash_part.strip() or None
    return parsed_query, parsed_hash


def _print_provenance(provenance: ProvenanceChain | None) -> None:
    if provenance is None:
        print("ProvenanceChain:")
        print("  Document: unknown")
        print("  Page: unknown")
        print("  Hash: unavailable")
        print("  BBox: [0.0, 0.0, 0.0, 0.0]")
        return

    print("ProvenanceChain:")
    print(f"  Document: {provenance.document_name}")
    print(f"  Page: {provenance.page_number}")
    print(f"  Hash: {provenance.content_hash}")
    print(f"  BBox: {provenance.bbox}")


def run_refinery(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    source_filename = os.path.basename(str(pdf_path))

    print("=" * 72)
    print(f"Doc Intelligence Refinery: {source_filename}")
    print("=" * 72)

    # 1) Extraction
    extractor = ExtractionRouter()
    extraction_result = extractor.process_document(str(pdf_path))
    print(f"Extraction complete: {len(extraction_result.ldus)} raw LDU(s)")

    # 2) Chunking
    chunker = SemanticChunker()
    raw_extractions = _to_raw_extractions(extraction_result.ldus, source_filename)
    ldus = chunker.process_segments(raw_extractions)

    # Preserve source filename on each LDU so provenance document_name is explicit.
    for ldu in ldus:
        object.__setattr__(ldu, "filename", source_filename)
        object.__setattr__(ldu, "document_name", source_filename)

    print(f"Chunking complete: {len(ldus)} validated LDU(s)")

    # 3) Indexing
    indexer = DocumentIndexer()
    page_index = indexer.build_index(ldus)
    print(f"Indexing complete: {len(page_index)} top-level page node(s)")

    # 4) Fact extraction into SQLite
    fact_extractor = FactTableExtractor(db_path=".refinery/facts.db")
    extracted_facts = fact_extractor.extract_facts_from_ldus(ldus)
    print(f"Fact extraction complete: {len(extracted_facts)} fact(s) stored in .refinery/facts.db")

    # 5) Vector storage (Chroma)
    vector_store = VectorStoreManager(db_path="data/vector_db", source_filename=source_filename)
    chunks = _ldus_to_chunks(ldus, filename=source_filename)
    vector_store.collection.delete(where={"filename": source_filename})
    vector_store.add_chunks(chunks)
    print(f"Vector storage complete: {len(chunks)} chunk(s) upserted")

    # 6) Query orchestration loop
    orchestrator = QueryOrchestrator(
        page_index=page_index,
        vector_store=VectorStoreManager(source_filename=source_filename),
        fact_extractor=fact_extractor,
    )

    print("\nInteractive Query Mode")
    print("Type a question, or `quit` to exit.")
    print("For strict hash verification, append `|hash=<content_hash>` to your query.")

    while True:
        user_input = input("\nQuery> ").strip()
        if not user_input:
            continue
        if user_input.casefold() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        query_text, expected_hash = _parse_query_and_hash(user_input)
        result_state = orchestrator.run(query=query_text, expected_hash=expected_hash)

        answer = str(result_state.get("final_response", "No answer generated."))
        provenance = result_state.get("final_provenance")

        print("\nAnswer")
        print("-" * 72)
        print(answer)
        print("\nEvidence")
        print("-" * 72)
        if isinstance(provenance, ProvenanceChain):
            _print_provenance(provenance)
        else:
            _print_provenance(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the end-to-end document intelligence refinery pipeline.")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/raw/Consumer Price Index May 2025.pdf"),
        help="Path to source PDF",
    )
    args = parser.parse_args()

    run_refinery(args.pdf)
