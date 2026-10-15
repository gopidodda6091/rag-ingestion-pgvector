# rag-ingestion-pgvector

![CI](https://github.com/gopidodda6091/rag-ingestion-pgvector/actions/workflows/ci.yml/badge.svg) ![License: MIT](https://img.shields.io/badge/license-MIT-blue)

The data-engineering half of retrieval-augmented generation: parse, chunk, redact PII, embed and **incrementally** index documents into Postgres/pgvector, then measure retrieval quality. No paid API key is required; LLM answer generation is an optional adapter.

> Fictional corpus: **Pecan Health Plans** member handbooks, policy documents and FAQs, written synthetically with planted PII for the redaction tests.

## Architecture
```mermaid
flowchart LR
  D[(docs/<br/>md · html · pdf)] --> P[parse.py]
  P --> H{content hash<br/>changed?}
  H -->|no| SKIP[skip]
  H -->|yes| CH[chunk.py<br/>heading-aware, overlap]
  CH --> PII[redact.py<br/>regex + rules<br/>audit log]
  PII --> EM[embed.py<br/>sentence-transformers]
  EM --> PG[(Postgres + pgvector<br/>chunks · embeddings · doc_versions)]
  PG --> HS[search.py<br/>BM25 + vector, reciprocal rank fusion]
  Q[(eval questions<br/>with gold chunks)] --> EV[evaluate.py<br/>recall@k · MRR]
  HS --> EV
  HS -.optional.-> LLM[answer adapter]
```

## Run it in three commands
```bash
make up      # Postgres + pgvector via docker compose
make demo    # ingest corpus, edit one doc, re-ingest (only it is re-embedded), run eval
make test    # pytest: chunking, PII redaction, idempotent re-ingest, retrieval metrics
```

## What it proves
- Incremental, idempotent ingestion keyed by document version and content hash
- PII controls in the pipeline, with an audit trail of what was redacted and why
- Hybrid retrieval and a measurable evaluation loop instead of "it looks right"

## Limitations
- Regex/rule-based PII detection is a baseline, not a compliance guarantee.
- Single-node Postgres; there is no ANN index tuning beyond HNSW defaults.
- Embeddings default to a deterministic hashing embedder (used by tests, CI and the demo). sentence-transformers all-MiniLM-L6-v2 is used only with `USE_ST_MODEL=1` after installing `requirements-model.txt`.
- The sample corpus is 11 short documents with 21 eval questions, so retrieval scores sit near the ceiling; they show the evaluation loop works, not real-world quality.
- Redaction runs on each section before it is split into overlapping windows, so a value cannot straddle a chunk boundary. The diagram shows the logical stages.
- PDF parsing reads text-layer PDFs only and detects headings heuristically. The bundled answer adapter is extractive (no LLM).

## About the author
Built by Gopi Dodda, Senior Data Engineer: [LinkedIn](https://www.linkedin.com/in/gopidodda6091/). Original work on synthetic data.
