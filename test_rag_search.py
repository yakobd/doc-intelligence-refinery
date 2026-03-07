from __future__ import annotations

import os
import shutil
from pathlib import Path

from src.agents.chunker import SemanticChunker
from src.agents.extractor import ExtractionRouter
from src.agents.indexer import DocumentIndexer
from src.engines.vector_store import VectorStoreManager
from src.models.document_schema import Chunk, LDU, PageIndexNode


def _print_results(title: str, results: list[dict], limit: int = 3) -> None:
    print(f"\n{title}")
    print("-" * 70)
    if not results:
        print("No results.")
        return

    for idx, result in enumerate(results[:limit], start=1):
        metadata = result.get("metadata", {}) or {}
        page_numbers = metadata.get("page_numbers", "")
        snippet = (result.get("document", "") or "").replace("\n", " ").strip()
        snippet = snippet[:180] + ("..." if len(snippet) > 180 else "")
        print(f"{idx}. id={result.get('id', 'N/A')}")
        print(f"   pages={page_numbers}")
        print(f"   distance={result.get('distance', 'N/A')}")
        print(f"   snippet={snippet}")


def _page_title_lookup(nodes: list[PageIndexNode]) -> dict[int, str]:
    page_titles: dict[int, str] = {}
    for node in nodes:
        for page in range(node.page_start, node.page_end + 1):
            if page not in page_titles:
                page_titles[page] = node.title
    return page_titles


def _ldus_to_chunks(ldus: list[LDU], page_titles: dict[int, str], filename: str) -> list[Chunk]:
    chunks: list[Chunk] = []

    for ldu in ldus:
        content = (ldu.content or "").strip()
        if not content:
            continue

        primary_page = ldu.page_refs[0] if ldu.page_refs else 1
        page_title = page_titles.get(primary_page, f"Page {primary_page}")
        token_count = len(content.split())

        chunks.append(
            Chunk(
                uid=f"{ldu.uid}-chunk",
                content=content,
                content_hash=ldu.content_hash,
                metadata={
                    "page_numbers": list(ldu.page_refs),
                    "parent_ldu_id": ldu.uid,
                    "unit_type": ldu.unit_type,
                    "title": page_title,
                    "filename": filename,
                    "content_hash": ldu.content_hash,
                },
                token_count=token_count,
            )
        )

    return chunks


def run_rag_search(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    refinery_dir = Path(".refinery")
    if refinery_dir.exists():
        shutil.rmtree(refinery_dir)
        print(f"Removed cache directory: {refinery_dir}")

    extractor = ExtractionRouter()
    chunker = SemanticChunker()
    indexer = DocumentIndexer()
    vector_store = VectorStoreManager()

    print("=" * 70)
    print(f"RAG SEARCH TEST: {pdf_path.name}")
    print("=" * 70)

    artifact_path = Path(".refinery/extractions/Consumer Price Index May 2025_extracted.json")
    if artifact_path.exists():
        os.remove(artifact_path)
        print(f"Removed stale extraction artifact: {artifact_path}")

    extraction_result = extractor.process_document(str(pdf_path))
    raw_extractions = [
        {
            "uid": ldu.uid,
            "content": ldu.content,
            "chunk_type": ldu.chunk_type,
            "page": ldu.page_refs[0] if ldu.page_refs else 1,
            "bounding_box": ldu.bounding_box,
            "parent_section": ldu.parent_section,
            "token_count": ldu.token_count,
            "child_chunks": list(ldu.child_chunks),
        }
        for ldu in extraction_result.ldus
    ]
    validated_ldus = chunker.process_segments(raw_extractions)
    page_index = indexer.build_index(validated_ldus)
    print("\nPageIndex titles:")
    for node in page_index:
        print(f"- {node.title}")
    page_titles = _page_title_lookup(page_index)

    all_chunks = _ldus_to_chunks(ldus=validated_ldus, page_titles=page_titles, filename=pdf_path.name)

    if not all_chunks:
        print("No chunks generated from this document. Exiting test.")
        return

    # Fresh re-ingestion: wipe previous vectors for this filename.
    vector_store.collection.delete(where={"filename": pdf_path.name})
    vector_store.add_chunks(all_chunks)

    query = "What are the price index changes?"
    print(f"\nQuery: {query}")

    # Step A: PageIndex traversal for relevant pages.
    # Uses hierarchical PageIndexNode traversal/scoring to choose relevant pages.
    page_filter = indexer.get_relevant_pages(query=query, nodes=page_index)

    print(f"Step A - Relevant pages from index: {page_filter if page_filter else 'None'}")

    # Step B: Vector search constrained to relevant pages.
    smart_results = vector_store.query(text=query, n_results=3, page_filter=page_filter)
    _print_results("Smart Results (Filtered by PageIndex)", smart_results)

    # Baseline vector search without page filtering.
    unfiltered_results = vector_store.query(text=query, n_results=3, page_filter=None)
    _print_results("Unfiltered Results (No page filter)", unfiltered_results)

    all_pages_in_doc = sorted({page for ldu in validated_ldus for page in ldu.page_refs})
    skipped_pages = sorted(set(all_pages_in_doc) - set(page_filter)) if page_filter else []

    print("\nComparison")
    print("-" * 70)
    print(f"Total LDUs: {len(validated_ldus)}")
    print(f"Total chunks indexed: {len(all_chunks)}")
    print(f"Pages in document: {all_pages_in_doc}")
    print(f"Pages selected by smart filter: {page_filter}")
    print(f"Pages skipped by smart filter: {len(skipped_pages)}")
    if skipped_pages:
        print(f"Skipped page numbers: {skipped_pages}")


if __name__ == "__main__":
    sample_pdf = Path("data/raw/Consumer Price Index May 2025.pdf")
    run_rag_search(sample_pdf)
