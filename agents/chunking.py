from langchain_community.document_loaders import DirectoryLoader,TextLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
from hybrid import HybridRetriever,HybridRetrieverWrapper
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever


embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", 
    model_kwargs={"device": "cpu"})

def load_n_chunk_data(data_path:str)->list[str]:
    loader=DirectoryLoader(
        data_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )
    
    raw_docs=loader.load()

    docs=[]

    for doc in raw_docs:
        text = doc.page_content

        pattern = r"\n(?=[A-Z][a-zA-Z ]+\n)"
        sections = re.split(pattern, text)

        for section in sections:
            if len(section.strip()) > 100:
                docs.append(
                    Document(
                        page_content=section.strip(),
                        metadata={"source": doc.metadata.get("source", "unknown")}
                    )
                )

    return docs

def build_hybrid_retriever(docs):

    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    dense_vectordb = FAISS.from_documents(docs, embedding)
    dense_retriever = dense_vectordb.as_retriever()

    sparse_retriever = BM25Retriever.from_documents(docs)
    sparse_retriever.k = 3

    hybrid = HybridRetriever(
        dense=dense_retriever,
        sparse=sparse_retriever,
        weights=(0.7, 0.3)
    )

    return HybridRetrieverWrapper(hybrid=hybrid)


    



