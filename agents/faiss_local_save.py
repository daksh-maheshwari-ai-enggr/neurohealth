import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from chunking import load_n_chunk_data,build_hybrid_retriever

DATA_PATH = "C:\orchestration\data"
INDEX_PATH = "rag/vector_store/faiss_index"

docs = load_n_chunk_data(DATA_PATH)

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vectordb = FAISS.from_documents(docs, embedding)

vectordb.save_local(INDEX_PATH)

print("FAISS index saved successfully.")

