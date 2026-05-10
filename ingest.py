"""
ingest.py — Document ingestion pipeline for the G-P RAG demo.

Pipeline stages:
1. Load PDFs from docs/
2. Extract text from each PDF
3. Chunk text into ~500-token segments with overlap
4. Embed each chunk via OpenAI text-embedding-3-small
5. Store chunks + embeddings + metadata in ChromaDB

Run once before query.py to populate the vector database.
"""

import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from pypdf import PdfReader
import tiktoken

# ───────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────

load_dotenv()  # Reads .env file, makes OPENAI_API_KEY available

# Paths
DOCS_DIR = Path("docs")           # Where PDFs live
DB_DIR = Path("data/chroma_db")   # Where ChromaDB persists to disk

# Chunking parameters
CHUNK_SIZE = 500       # Target tokens per chunk
CHUNK_OVERLAP = 75     # Tokens of overlap between adjacent chunks (~15%)

# Models
EMBEDDING_MODEL = "text-embedding-3-small"   # 1536 dimensions, $0.02/1M tokens
EMBEDDING_DIM = 1536

# Collection name (a "collection" in Chroma is like a table in SQL)
COLLECTION_NAME = "compliance_docs"

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

# Tiktoken encoder — used to count tokens accurately
# cl100k_base is what OpenAI's models use
encoding = tiktoken.get_encoding("cl100k_base")

print(f"✓ Initialized. Will ingest from {DOCS_DIR.resolve()}")
print(f"✓ ChromaDB will persist to {DB_DIR.resolve()}")