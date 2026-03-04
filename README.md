# Document Intelligence Refinery (10 Academy)

## Project Overview

This project implements an automated **Document Intelligence Refinery** pipeline for 10 Academy. It triages incoming PDFs and routes each document to the most appropriate extraction strategy:

- **Strategy A (FastText)**: Fast, text-first extraction for native digital and simple-layout documents.
- **Strategy B (Docling)**: Layout-aware extraction for multi-column and table-heavy documents.
- **Strategy C (Vision/OCR)**: Vision-assisted extraction for scanned/image-based documents with budget guardrails.

The pipeline includes profiling, strategy routing, normalized output schemas, and extraction ledger tracking for auditability.

## Setup Instructions

### 1) Create and activate the Conda environment

```bash
conda create -n refinery python=3.10 -y
conda activate refinery
```

### 2) Install project dependencies

```bash
pip install -r requirements.txt
```

If your environment is missing strategy-specific packages, install them with:

```bash
pip install pydantic docling httpx pdf2image pytest
```

## Usage

Run the full end-to-end pipeline on files in `data/raw`:

```bash
python -m tests.test_end_to_end
```

This command performs triage, routes documents through Strategy A/B/C, extracts structured outputs, and updates the extraction ledger.

## Project Structure

- **`src/agents/`**
  - Contains orchestration and decision logic.
  - `triage.py` profiles documents (origin, layout complexity, domain hint, estimated cost).
  - `extractor.py` routes each file to the proper strategy and writes ledger records.

- **`.refinery/profiles/`**
  - Stores generated profiling artifacts (JSON) used for traceability and rubric evidence.
  - Supports reproducibility of routing decisions.

- **`rubric/`**
  - Contains assignment/rubric-aligned requirements, checkpoints, and deliverable guidance.
  - Serves as the reference for compliance and submission validation.

## Evidence (Rubric Alignment)

The implemented pipeline produces rubric-required evidence outputs, including:

- **12 JSON profile artifacts** generated under `.refinery/profiles/`.
- A persistent extraction ledger at `.refinery/extraction_ledger.jsonl` that records timestamp, filename, strategy used, average confidence, and estimated extraction cost.

These artifacts provide auditable proof of triage decisions and extraction execution quality.
