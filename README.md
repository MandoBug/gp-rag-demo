# G-P RAG Demo — HR/Labor Compliance Q&A

A retrieval-augmented generation (RAG) system that answers HR and labor compliance questions using a corpus of official government documents (US DOL fact sheets, IRS publications, California state law, Irish employment law). Built to mirror the architectural pattern behind compliance assistants like G-P's Gia.

The point of this project: minimize hallucination in a high-stakes domain by **grounding LLM responses in retrieved documents instead of relying on parametric memory.**

---

## Why this matters

LLMs trained on next-token prediction are optimized for plausibility, not truth. In casual use, hallucination is a UX problem. In HR/legal compliance, a wrong answer creates real legal liability for the customer — labor law violations, tax penalties, unenforceable contracts.

This system uses RAG to ground every answer in retrieved source documents and cite the filename used. The LLM is reading, not remembering.

---

## Architecture

```
INGESTION (run once)
─────────────────────
PDFs in docs/
   │
   ├── pypdf: extract text
   │
   ├── tiktoken: chunk by tokens (500 tokens, 75 overlap)
   │
   ├── OpenAI text-embedding-3-small: 1536-dim vectors
   │
   └── ChromaDB (cosine similarity, HNSW index)
                                                    │
                                                    ▼
QUERY (run live)                          data/chroma_db/
─────────────────                                   │
User question                                       │
   │                                                │
   ├── Embed question (same model)                  │
   │                                                │
   ├── Retrieve top-5 chunks ──────────────────────┘
   │
   ├── Build prompt: system + retrieved context + question
   │
   ├── Call gpt-4o-mini (temperature=0)
   │
   └── Return answer + cited sources
```

---

## Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Embedding model | OpenAI `text-embedding-3-small` | 1536 dims, $0.02 / 1M tokens, strong baseline |
| Vector DB | ChromaDB (persistent, local) | Zero-setup, HNSW indexing, cosine similarity |
| LLM | OpenAI `gpt-4o-mini` | $0.15 / 1M input tokens; sufficient quality for grounded summarization |
| PDF extraction | `pypdf` | Standard, no external dependencies |
| Tokenizer | `tiktoken` (cl100k_base) | Matches OpenAI's actual tokenization |

---

## Design decisions

### Chunking — 500 tokens, 75-token overlap (~15%)

Token-based, not character-based. Embedding models have token limits, not character limits, and tokens give predictable cost per chunk. The 500/75 split is a baseline sweet spot: large enough to carry paragraph-level context, small enough that retrieval stays focused. Overlap exists so a key sentence at a chunk boundary isn't lost between two adjacent chunks.

The tradeoff: more overlap means each text region gets embedded multiple times, increasing storage and embedding cost. A production upgrade would be a recursive splitter that respects paragraph and sentence boundaries — better chunk coherence at the cost of more complex code.

### Cosine similarity over L2 distance

Configured ChromaDB to use cosine similarity instead of the default L2. Embeddings encode meaning in their direction; magnitude is mostly noise. Cosine measures angle (ignoring magnitude), so two vectors pointing the same way are treated as similar regardless of their length. L2 conflates magnitude with similarity and produces worse retrieval for embeddings.

### `text-embedding-3-small` over `large`

The `large` model (3072 dims) is higher quality but costs 6x more and uses 2x the storage per vector. For general-purpose retrieval over compliance documents, `small` is plenty — there's no evidence the quality difference matters at this scale.

### `gpt-4o-mini` over `gpt-4o`

For RAG, the LLM mostly summarizes provided context rather than reasoning from scratch. `gpt-4o-mini` does this job at ~17x the cost-efficiency of `gpt-4o`. Reserve the bigger model for tasks that need deep reasoning beyond what's in the retrieved chunks.

### `temperature=0`

Compliance answers must be consistent. Same question + same retrieved context should always produce the same answer. Higher temperature introduces randomness, which is the opposite of what you want for legal/HR Q&A.

### System prompt — anti-hallucination by construction

The system prompt explicitly instructs the model to:
1. Use only the provided context — no outside knowledge
2. Say "I don't know" when context is insufficient (most important rule — without this, the model fills gaps with plausible nonsense)
3. Cite source filenames in brackets — verifiability
4. Stay on-topic — politely redirect off-topic questions

The "permit 'I don't know'" rule is the single most important hallucination defense at the prompt layer.

---

## Hallucination mitigation pyramid

This system implements layers 1, 2, and 5 of the standard hallucination mitigation stack:

1. **Grounding via RAG** ✅ — every answer is generated from retrieved chunks
2. **Source citations** ✅ — every answer includes the filename it pulled from
3. Eval harnesses — *not implemented; production upgrade*
4. Confidence calibration — *partial via prompt; not measured*
5. Constrained generation via system prompt ✅ — scope-limited and grounding-required
6. Human-in-the-loop — *not implemented; relevant for high-stakes deployments*

---

## How to run

### Prerequisites

- Python 3.11+
- OpenAI API key (set in `.env` as `OPENAI_API_KEY=...`)

### Setup

```bash
# Clone
git clone https://github.com/MandoBug/gp-rag-demo.git
cd gp-rag-demo

# Virtual env
python -m venv venv
source venv/bin/activate          # macOS/Linux
# OR: venv\Scripts\activate       # Windows

# Install
pip install -r requirements.txt

# Add your OpenAI key to .env
echo "OPENAI_API_KEY=sk-..." > .env
```

### Ingest documents

Place PDFs in `docs/`, then:

```bash
python ingest.py
```

This loads, chunks, embeds, and persists everything to `data/chroma_db/`. Estimated cost for the included corpus (~432 chunks, ~213K tokens): under one cent.

### Query

```bash
python query.py
```

Interactive prompt. Type a compliance question, get an answer with citations. Example:

```
> What is overtime pay under the FLSA?

Overtime pay under the FLSA is the premium pay employers must provide to
covered, nonexempt employees for hours worked in excess of 40 in a workweek,
at a rate not less than time and one-half the regular rate of pay
[US_DOL_FactSheet_23_Overtime_Pay.pdf].
```

---

## Sample questions to try

- *What is overtime pay under the FLSA?* → returns FLSA overtime requirements
- *What are the rules for paying interns under US law?* → returns the 7-factor primary beneficiary test
- *What rights do domestic workers have in Ireland?* → returns Irish-specific employment rights
- *What is the minimum wage for healthcare workers in California in 2026?* → returns tiered wage by facility type
- *What's the minimum wage in Brazil?* → returns "I don't have enough context" (no Brazilian docs in corpus — proves grounding works)

---

## What I'd improve next

- **Recursive chunking** — respect paragraph and sentence boundaries instead of cutting mid-sentence at token windows
- **Eval harness** — a fixed set of question-expected-source pairs, running automatically to catch retrieval regressions
- **Hybrid search** — combine vector similarity with keyword/BM25 search; helps with exact-match queries (statute numbers, dollar figures) where pure semantic search underperforms
- **Re-ranking** — use a cross-encoder model on the top-20 retrieved chunks to re-rank to top-5 by relevance, often improves precision significantly
- **Chunk-level confidence** — surface retrieval distance scores in the UI so users can self-assess answer reliability
- **Hosted vector DB** — migrate from local ChromaDB to a managed service (Pinecone, Qdrant Cloud) for production deployment
- **Multi-language support** — re-embed with a multilingual model for non-English compliance docs

---

## Corpus

The included corpus covers cross-border employment compliance:

- **US Federal:** DOL fact sheets (overtime, FMLA, internships, employment relationship), IRS publications (employer tax guides, fringe benefits)
- **California state:** minimum wage rules, healthcare worker wage supplements
- **Ireland:** general employment law, domestic worker rights

Total: 11 documents, ~213K tokens, 432 chunks indexed.

---

## Project structure

```
gp-rag-demo/
├── docs/                  # Source PDFs (committed)
├── data/                  # ChromaDB persistence (gitignored)
├── ingest.py              # Ingestion pipeline (load → chunk → embed → store)
├── query.py               # Query pipeline (embed → retrieve → generate)
├── app.py                 # Streamlit UI (optional)
├── requirements.txt
├── .env                   # OPENAI_API_KEY (gitignored)
└── README.md
```

---

## Author

[Armando Tamayo](https://github.com/MandoBug) — built as a learning exercise for an internship interview at G-P, focused on understanding the architectural patterns behind compliance-focused LLM products like Gia.