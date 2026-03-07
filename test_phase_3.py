from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.extractor import ExtractionRouter
from src.agents.triage import TriageAgent
from src.engines.chunking import Chunk, ChunkingEngine
from src.engines.indexer import DocumentIndexer
from src.models.document_schema import PageIndexNode


def print_index_tree(nodes: list[PageIndexNode], level: int = 0) -> None:
    indent = "  " * level
    for node in nodes:
        print(f"{indent}- {node.title}")
        if node.children:
            print_index_tree(node.children, level + 1)


def verify_chunk_hashes(chunks: list[Chunk], hash_algorithm: str = "sha256") -> tuple[bool, list[str]]:
    invalid_chunk_ids: list[str] = []
    for chunk in chunks:
        expected_hash = ChunkingEngine.compute_hash(chunk.content, hash_algorithm)
        if chunk.content_hash != expected_hash:
            invalid_chunk_ids.append(chunk.uid)
    return len(invalid_chunk_ids) == 0, invalid_chunk_ids


def run_phase_3(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    triage = TriageAgent()
    extractor = ExtractionRouter()
    chunking_engine = ChunkingEngine()
    indexer = DocumentIndexer()

    print("=" * 60)
    print(f"PHASE 3 TEST: {pdf_path.name}")
    print("=" * 60)

    profile = triage.profile_document(str(pdf_path))
    print("[1/4] Triage complete")
    print(f"  Origin: {profile.origin_type.value}")
    print(f"  Layout: {profile.layout_complexity.value}")
    print(f"  Strategy: {profile.selected_strategy.value}")
    print(f"  Estimated chars: {profile.estimated_chars}")

    extraction_result = extractor.process_document(str(pdf_path))
    ldus = extraction_result.ldus
    print("[2/4] Extraction complete")
    print(f"  LDUs extracted: {len(ldus)}")

    all_chunks: list[Chunk] = []
    for ldu in ldus:
        all_chunks.extend(chunking_engine.chunk_ldu(ldu))

    print("[3/4] Chunking complete")
    print(f"  Total chunks generated: {len(all_chunks)}")

    page_index = indexer.build_index(ldus)
    print("[4/4] Indexing complete")

    hash_valid, invalid_chunk_ids = verify_chunk_hashes(all_chunks)

    print("\nSummary")
    print("-" * 60)
    print(f"Total LDUs found: {len(ldus)}")
    print(f"Total Chunks generated: {len(all_chunks)}")
    print(f"All chunk hashes valid: {'YES' if hash_valid else 'NO'}")
    if not hash_valid:
        print("Invalid chunk IDs:")
        for chunk_id in invalid_chunk_ids:
            print(f"  - {chunk_id}")

    print("\nPageIndex hierarchy")
    print("-" * 60)
    print_index_tree(page_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 integration smoke test")
    parser.add_argument(
        "pdf",
        nargs="?",
        default="data/raw/Consumer Price Index May 2025.pdf",
        help="Path to a sample PDF",
    )
    args = parser.parse_args()

    run_phase_3(Path(args.pdf))


if __name__ == "__main__":
    main()
