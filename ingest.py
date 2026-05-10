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

# ───────────────────────────────────────────────────────────────
# Stage 1: Load PDFs and extract text
# ───────────────────────────────────────────────────────────────

def load_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file.
    
    pypdf handles text-based PDFs reliably. Scanned/image-only PDFs
    would need OCR (e.g. tesseract), which we're not using here.
    Returns concatenated text from all pages, separated by newlines.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""  # extract_text() can return None
        pages.append(text)
    return "\n".join(pages)


def load_all_pdfs(docs_dir: Path) -> List[Dict]:
    """
    Load every PDF in docs_dir and return a list of documents.
    Each document is a dict with the filename and its full text.
    """
    documents = []
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠ No PDFs found in {docs_dir}")
        return documents
    
    print(f"\nFound {len(pdf_files)} PDFs. Loading...")
    
    for pdf_path in pdf_files:
        try:
            text = load_pdf(pdf_path)
            documents.append({
                "filename": pdf_path.name,
                "text": text,
                "char_count": len(text),
            })
            print(f"  ✓ {pdf_path.name}  ({len(text):,} chars)")
        except Exception as e:
            print(f"  ✗ {pdf_path.name}  — error: {e}")
    
    return documents
# ───────────────────────────────────────────────────────────────
# Stage 2: Chunk text into overlapping segments
# ───────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source_filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Split text into overlapping token-based chunks.
    
    Strategy: tokenize the full text with tiktoken, then slide a window
    of `chunk_size` tokens across, advancing by `chunk_size - chunk_overlap`
    each step. Each window is decoded back to text and saved as a chunk.
    
    Why token-based and not character-based?
        Embedding models have token limits, not character limits.
        A 500-token chunk has predictable embedding cost and stays
        within model limits regardless of vocabulary.
    
    Why the window approach instead of splitting on paragraphs?
        Simple, predictable, no edge cases with weird PDF formatting.
        Production systems often use recursive splitters (paragraph →
        sentence → word) to respect natural boundaries — this is the
        baseline version of that idea.
    """
    # Encode the full document into tokens
    tokens = encoding.encode(text)
    total_tokens = len(tokens)
    
    if total_tokens == 0:
        return []
    
    # Stride = how far we move the window each iteration
    stride = chunk_size - chunk_overlap
    
    chunks = []
    chunk_index = 0
    start = 0
    
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text_str = encoding.decode(chunk_tokens)
        
        chunks.append({
            "chunk_id": f"{source_filename}::chunk_{chunk_index}",
            "text": chunk_text_str,
            "source": source_filename,
            "chunk_index": chunk_index,
            "token_count": len(chunk_tokens),
            "token_start": start,
            "token_end": end,
        })
        
        chunk_index += 1
        
        # If we've reached the end, stop. Otherwise advance by stride.
        if end >= total_tokens:
            break
        start += stride
    
    return chunks


def chunk_all_documents(documents: List[Dict]) -> List[Dict]:
    """
    Apply chunking to every document, return a flat list of all chunks.
    """
    all_chunks = []
    
    print(f"\nChunking {len(documents)} documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    
    for doc in documents:
        chunks = chunk_text(doc["text"], doc["filename"])
        all_chunks.extend(chunks)
        print(f"  ✓ {doc['filename']:60s} → {len(chunks):4d} chunks")
    
    return all_chunks


# ───────────────────────────────────────────────────────────────
# Stage 3: Embed chunks via OpenAI
# ───────────────────────────────────────────────────────────────

def embed_chunks(
    chunks: List[Dict],
    batch_size: int = 100,
) -> List[Dict]:
    """
    Generate embeddings for every chunk, attach the vectors to the chunk dicts.
    
    Why batch?
        Each API call has overhead. Sending 100 texts in one call is much
        faster than 100 separate calls. OpenAI accepts up to 2048 inputs
        per request — we cap at 100 for safety + better progress visibility.
    
    Why the same model used at query time?
        Embeddings only make sense within the model that produced them.
        Different models = different vector spaces = meaningless similarity.
    """
    if not chunks:
        return chunks
    
    print(f"\nEmbedding {len(chunks)} chunks via {EMBEDDING_MODEL}...")
    print(f"  Batch size: {batch_size}")
    
    total_batches = (len(chunks) + batch_size - 1) // batch_size  # ceiling division
    
    for batch_num, start in enumerate(range(0, len(chunks), batch_size), 1):
        batch = chunks[start:start + batch_size]
        texts = [c["text"] for c in batch]
        
        try:
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
        except Exception as e:
            print(f"  ✗ Batch {batch_num}/{total_batches} failed: {e}")
            raise  # Halt — we don't want a partially embedded DB
        
        # Attach embeddings back to the original chunk dicts
        for chunk, embedding_data in zip(batch, response.data):
            chunk["embedding"] = embedding_data.embedding
        
        print(f"  ✓ Batch {batch_num}/{total_batches}  ({len(batch)} chunks embedded)")
    
    return chunks
# ───────────────────────────────────────────────────────────────
# Main execution
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("G-P RAG Demo — Document Ingestion")
    print(f"{'='*60}\n")
    
    print(f"Docs directory: {DOCS_DIR.resolve()}")
    print(f"Vector DB:      {DB_DIR.resolve()}")
    
    # Stage 1: Load PDFs
    documents = load_all_pdfs(DOCS_DIR)
    if not documents:
        print("\nNo documents loaded. Add PDFs to docs/ and try again.")
        exit(1)
    total_chars = sum(d["char_count"] for d in documents)
    print(f"\n✓ Loaded {len(documents)} documents, {total_chars:,} total characters")
    
    # Stage 2: Chunk
    chunks = chunk_all_documents(documents)
    total_tokens = sum(c["token_count"] for c in chunks)
    print(f"\n✓ Created {len(chunks)} chunks, {total_tokens:,} total tokens")
    print(f"  Estimated embedding cost: ${total_tokens * 0.02 / 1_000_000:.4f}")
    
    # Stage 3: Embed
    chunks = embed_chunks(chunks)
    
    # Sanity check — confirm every chunk has an embedding of the right size
    sample = chunks[0]
    assert "embedding" in sample, "First chunk missing embedding!"
    assert len(sample["embedding"]) == EMBEDDING_DIM, (
        f"Wrong embedding dim: {len(sample['embedding'])} (expected {EMBEDDING_DIM})"
    )
    print(f"\n✓ All {len(chunks)} chunks embedded ({EMBEDDING_DIM} dims each)")