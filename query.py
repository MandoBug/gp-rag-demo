"""
query.py — RAG query pipeline for the G-P compliance demo.

Pipeline (run live, per question):
    1. Embed user question
    2. Retrieve top-k similar chunks from ChromaDB
    3. Build prompt with retrieved context
    4. Call GPT-4o-mini for grounded answer
    5. Display answer with citations
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# ───────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────

load_dotenv()

DB_DIR = "data/chroma_db"
COLLECTION_NAME = "compliance_docs"
EMBEDDING_MODEL = "text-embedding-3-small"   # MUST match ingest.py
LLM_MODEL = "gpt-4o-mini"                    # Cheap + sufficient for this
TOP_K = 5                                    # Chunks to retrieve per query

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)


# ───────────────────────────────────────────────────────────────
# Stage 1: Embed query and retrieve relevant chunks
# ───────────────────────────────────────────────────────────────

def retrieve(question: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Embed the question and retrieve top-k most similar chunks.
    
    Each result contains:
        - text:        chunk content (for prompt + display)
        - source:      filename (for citation)
        - chunk_index: position within source (for debugging)
        - distance:    cosine distance (lower = more similar)
    """
    # Embed the question. CRITICAL: same model as ingestion.
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
    )
    query_embedding = response.data[0].embedding
    
    # Search the vector DB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    
    # Flatten Chroma's nested response into a clean list
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": results["distances"][0][i],
        })
    
    return chunks


# ───────────────────────────────────────────────────────────────
# Stage 2: Build prompt and call the LLM
# ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an HR and labor compliance assistant. \
Your job is to answer questions using ONLY the context provided below. The context contains \
excerpts from official compliance documents (US Department of Labor fact sheets, IRS \
publications, Irish employment law, California state law).

RULES:
1. Answer using ONLY the provided context. Do NOT use outside knowledge.
2. If the context doesn't contain enough information to answer, say so explicitly. \
Do NOT guess or fabricate.
3. Cite sources inline by filename in square brackets, e.g. [US_DOL_FactSheet_23_Overtime_Pay.pdf].
4. Be concise and direct. Compliance answers must be precise.
5. If the user asks something outside HR/labor compliance, politely redirect them."""


def build_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Construct the user message: retrieved context + question.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i}: {chunk['source']}]\n{chunk['text']}"
        )
    
    context = "\n\n---\n\n".join(context_blocks)
    
    return f"""CONTEXT:
{context}

QUESTION: {question}

Answer the question using only the context above. Cite sources by filename in brackets."""


def generate_answer(question: str, chunks: List[Dict]) -> str:
    """
    Call GPT-4o-mini with the retrieval-augmented prompt.
    
    temperature=0.0 makes output deterministic — same question + same context
    produces the same answer. Critical for compliance work where consistency matters.
    """
    user_prompt = build_prompt(question, chunks)
    
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    
    return response.choices[0].message.content


# ───────────────────────────────────────────────────────────────
# Pipeline runner + interactive loop
# ───────────────────────────────────────────────────────────────

def answer_question(question: str, verbose: bool = True) -> None:
    """
    Full RAG pipeline: retrieve → generate → display.
    """
    print(f"\n{'='*70}")
    print(f"Question: {question}")
    print(f"{'='*70}")
    
    # Retrieve
    chunks = retrieve(question)
    
    if verbose:
        print(f"\nRetrieved {len(chunks)} chunks (top-k = {TOP_K}):")
        for i, chunk in enumerate(chunks, 1):
            preview = chunk["text"][:100].replace("\n", " ")
            print(f"  [{i}] {chunk['source']:60s} dist={chunk['distance']:.4f}")
            print(f"      \"{preview}...\"")
    
    # Generate
    print(f"\n{'─'*70}")
    print("Answer:")
    print(f"{'─'*70}")
    answer = generate_answer(question, chunks)
    print(answer)
    print()


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("G-P RAG Demo — Compliance Q&A")
    print(f"{'='*70}")
    print(f"Collection: {COLLECTION_NAME} ({collection.count()} chunks)")
    print(f"Type a question, or 'quit' to exit.\n")
    
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        
        try:
            answer_question(question)
        except Exception as e:
            print(f"Error: {e}\n")