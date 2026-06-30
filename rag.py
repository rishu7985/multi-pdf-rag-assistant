import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer,CrossEncoder
import google.generativeai as genai
from dotenv import load_dotenv
import os
import streamlit as st
from rank_bm25 import BM25Okapi

# ---------------- GLOBAL STATE ----------------
index = None
chunks = None
bm25 = None
tokenized_chunks = None

# ---------------- ENV ----------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# ---------------- MODELS ----------------
@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel("gemini-2.5-flash")

gemini_model = load_gemini_model()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = load_reranker()


# ---------------- LOAD INDEX ----------------
def load_index():
    global index, chunks, bm25, tokenized_chunks

    if os.path.exists("faiss_index.bin") and os.path.exists("chunks.pkl"):

        index = faiss.read_index("faiss_index.bin")

        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        # BM25 build
        tokenized_chunks = [
            chunk["text"].lower().split()
            for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_chunks)

        return True

    return False


# ---------------- QUERY REWRITE ----------------
def rewrite_query(query, history):
    conversation = ""

    for chat in history:
        conversation += f"""
User: {chat['question']}
Assistant: {chat['answer']}
"""

    prompt = f"""
Rewrite the user's latest question into a standalone question.

Conversation:
{conversation}

Latest Question:
{query}

Only output the rewritten question.
"""

    response = gemini_model.generate_content(prompt)

    try:
        return response.text.strip()
    except:
        return query


# ---------------- RAG CORE ----------------
def ask_rag(query, history=None, k=5, selected_pdfs=None, streaming=False):

    global index, chunks, bm25

    if index is None or chunks is None:
        if not load_index():
            return {
                "answer": "Please build the index first.",
                "pages": [],
                "sources": [],
                "retrieved_chunks": []
            }

    if history is None:
        history = []

    if selected_pdfs is None:
        selected_pdfs = []

    search_query = query

    if history:
        search_query = rewrite_query(query, history)

    # ---------------- EMBEDDING ----------------
    query_embedding = model.encode(
        [search_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    D, I = index.search(query_embedding, 100)

    # ---------------- BM25 ----------------
    faiss_candidates = {}

    for score, idx in zip(D[0], I[0]):
        if idx == -1 or idx >= len(chunks):
            continue

        faiss_candidates[idx] = {
            "faiss_score": float(score),
            "bm25_score": 0.0
        }

    if bm25 is not None:
        query_tokens = search_query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:100]
    else:
        bm25_scores = np.zeros(len(chunks))
        bm25_top_idx = []

    for idx in bm25_top_idx:
        if idx >= len(chunks):
            continue

        if idx not in faiss_candidates:
            faiss_candidates[idx] = {
                "faiss_score": 0.0,
                "bm25_score": float(bm25_scores[idx])
            }
        else:
            faiss_candidates[idx]["bm25_score"] = float(bm25_scores[idx])

    # ---------------- MERGE ----------------
    candidates = []

    for idx, scores in faiss_candidates.items():

        chunk = chunks[idx]

        if selected_pdfs and chunk["source"] not in selected_pdfs:
            continue

        final_score = (
            0.6 * scores["faiss_score"] +
            0.4 * scores["bm25_score"]
        )

        citation = f"{chunk['source']} (Page {chunk['page']})"

        candidates.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "source": chunk["source"],
            "citation": citation,
            "score": final_score
        })

    pairs = [
        (search_query,candidate["text"]) for candidate in candidates
    ]
    rerank_scores = reranker.predict(pairs)
    for candidate,score in zip(candidates,rerank_scores):
        candidate["rerank_score"] = float(score)

    candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    retrieved_chunks = candidates[:k]


    if len(retrieved_chunks) == 0:
        return {
            "answer": "No relevant information found.",
            "pages": [],
            "sources": [],
            "retrieved_chunks": []
        }

    # ---------------- METADATA ----------------
    citations = sorted(set(c["citation"] for c in retrieved_chunks))
    context = "\n\n".join(c["text"] for c in retrieved_chunks)

    pages = sorted(set(c["page"] for c in retrieved_chunks))
    sources = sorted(set(c["source"] for c in retrieved_chunks))

    # ---------------- PROMPT ----------------
    conversation = ""
    for chat in history:
        conversation += f"""
User: {chat['question']}
Assistant: {chat['answer']}
"""

    prompt = f"""
You are a strict RAG-based assistant.

Rules:
1. Answer ONLY using the provided context.
2. If context is insufficient, say:
   "I could not find this information in the document."
3. Do NOT use outside knowledge.

Context:
{context}

Conversation:
{conversation}

Question:
{query}

Citations available:
{citations}

Answer:
"""

    # ---------------- STREAMING ----------------
    if streaming:
        response = gemini_model.generate_content(prompt, stream=True)

        def stream_generator():
            for chunk in response:
                if chunk.text:
                    yield chunk.text

        return stream_generator()

    # ---------------- NORMAL RESPONSE ----------------
    response = gemini_model.generate_content(prompt)

    answer = ""
    for chunk in response:
        if chunk.text:
            answer += chunk.text

    return {
        "answer": answer,
        "pages": pages,
        "sources": sources,
        "rewritten_query": search_query,
        "citations": citations,
        "retrieved_chunks": retrieved_chunks
    }