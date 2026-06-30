import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from sentence_transformers.util import normalize_embeddings

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_pdfs(pdf_folder):
    documents =[]
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder,file)
            reader = PdfReader(pdf_path)
            for page_number,page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                documents.append(
                    {
                        "page": page_number,
                        "text": text,
                        "source": file
                    }
                )
    return documents

def chunk_documents(documents,chunk_size=500,overlap=100):
    chunks = []
    for document in documents:
        text = document["text"]
        for st in range(0,len(text),chunk_size-overlap):
            chunk = text[st:st+chunk_size]
            if len(chunk.strip())<100:
                continue
            chunks.append(
                {
                    "text": chunk,
                    "source": document["source"],
                    "page": document["page"]
                }
            )
    return chunks

def generate_embeddings(chunks):
    chunks_texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(chunks_texts,
                    normalize_embeddings = True
                ).astype(np.float32)

    for i in range(len(chunks)):
        chunks[i]["embedding"] = embeddings[i]

    return chunks


def build_faiss_index(chunks):
    embeddings = np.array([chunk["embedding"] for chunk in chunks],
                          dtype = np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index

def save_index(index,chunks):
    faiss.write_index(index,"faiss_index.bin")

    with open("chunks.pkl","wb") as f:
        pickle.dump(chunks,f)
