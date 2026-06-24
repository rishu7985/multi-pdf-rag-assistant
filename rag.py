import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

gemini_model = genai.GenerativeModel("gemini-2.5-flash")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

index = faiss.read_index("faiss_index.bin")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


# %%
def ask_rag(query):
    query_embedding = model.encode(
        [query]
    ).astype(np.float32)

    D, I = index.search(query_embedding, k=5)

    context = "\n\n".join(
        chunks[idx]["text"] for idx in I[0]
    )

    pages = sorted(
        set(chunks[idx]["page"] for idx in I[0])
    )

    sources = sorted(
        set(chunks[idx]["source"] for idx in I[0])
    )
    prompt = f"""
    You are a helpful assistant.

    Answer ONLY from the provided context.

    If the answer is not present in the context, say:

    "I could not find this information in the document."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = gemini_model.generate_content(
        prompt
    )

    return {
        "answer": response.text,
        "pages": pages,
        "sources": sources
    }
