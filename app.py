"""
app.py — Streamlit UI for the G-P RAG compliance demo.

Wraps query.py's pipeline in a clean web interface for live demos.
Run with: streamlit run app.py
"""

import time
import streamlit as st
from query import retrieve, generate_answer, collection, TOP_K

# ───────────────────────────────────────────────────────────────
# Page setup
# ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="G-P RAG Compliance Demo",
    page_icon="📋",
    layout="centered",
)

# ───────────────────────────────────────────────────────────────
# Header
# ───────────────────────────────────────────────────────────────

st.title("HR & Labor Compliance Q&A")
st.caption(
    f"RAG-grounded answers from {collection.count()} chunks across "
    "US DOL, IRS, California, and Ireland compliance documents."
)

# ───────────────────────────────────────────────────────────────
# Sample questions sidebar
# ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Sample questions")
    samples = [
        "What is overtime pay under the FLSA?",
        "What are the rules for paying interns under US law?",
        "What rights do domestic workers have in Ireland?",
        "What is the minimum wage for healthcare workers in California in 2026?",
        "What's the minimum wage in Brazil?",
    ]
    for q in samples:
        if st.button(q, use_container_width=True, key=f"sample_{q}"):
            st.session_state["question"] = q

    st.divider()
    st.caption(
        "**Tip:** the Brazil question demonstrates hallucination defense — "
        "the system has no Brazilian docs and will say so explicitly instead "
        "of inventing an answer."
    )

# ───────────────────────────────────────────────────────────────
# Question input
# ───────────────────────────────────────────────────────────────

question = st.text_input(
    "Ask a compliance question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What is overtime pay under the FLSA?",
    key="question_input",
)

ask = st.button("Ask", type="primary", disabled=not question.strip())

# ───────────────────────────────────────────────────────────────
# Answer pipeline
# ───────────────────────────────────────────────────────────────

if ask and question.strip():
    # Stage 1: Retrieve
    with st.spinner("Retrieving relevant context..."):
        t0 = time.time()
        chunks = retrieve(question)
        retrieval_time = time.time() - t0

    # Stage 2: Generate
    with st.spinner("Generating answer..."):
        t1 = time.time()
        answer = generate_answer(question, chunks)
        generation_time = time.time() - t1

    total_time = retrieval_time + generation_time

    # ── Answer ──
    st.subheader("Answer")
    st.write(answer)

    # ── Cited sources ──
    cited_files = sorted({c["source"] for c in chunks})
    st.subheader("Sources used")
    for src in cited_files:
        st.markdown(f"- `{src}`")

    # ── Retrieval transparency (collapsible) ──
    with st.expander(f"Show retrieved chunks (top-{TOP_K})"):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(
                f"**[{i}] {chunk['source']}** "
                f"(distance: `{chunk['distance']:.4f}`)"
            )
            preview = chunk["text"][:500].replace("\n", " ")
            st.caption(f'"{preview}..."')
            st.divider()

    # ── Timing ──
    st.caption(
        f"⏱ Retrieval: {retrieval_time:.2f}s · "
        f"Generation: {generation_time:.2f}s · "
        f"Total: {total_time:.2f}s"
    )

# ───────────────────────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built with OpenAI `text-embedding-3-small`, `gpt-4o-mini`, and ChromaDB. "
    "Architecture mirrors compliance-focused RAG systems like G-P's Gia."
)