from collections import defaultdict
from langchain_core.retrievers import BaseRetriever


class HybridRetriever:
    def __init__(self, dense, sparse, weights=(0.7, 0.3), k=60):
        self.dense = dense
        self.sparse = sparse
        self.w_dense, self.w_sparse = weights
        self.k = k

    def _rrf(self, ranked_lists):
        scores = defaultdict(float)

        for weight, docs in ranked_lists:
            for rank, doc in enumerate(docs):
                scores[doc.page_content] += weight * (1 / (self.k + rank))

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [doc for doc, _ in ranked]

    def get_relevant_documents(self, query, top_k=5):
        dense_docs = self.dense.invoke(query)
        sparse_docs = self.sparse.invoke(query)

        ranked = self._rrf([
            (self.w_dense, dense_docs),
            (self.w_sparse, sparse_docs)
        ])

        mapping = {d.page_content: d for d in dense_docs + sparse_docs}
        return [mapping[c] for c in ranked[:top_k]]


# ADD THIS BELOW
from langchain_core.retrievers import BaseRetriever
from typing import Any

class HybridRetrieverWrapper(BaseRetriever):
    hybrid: Any   # 🔥 Pydantic field declare karna mandatory

    def _get_relevant_documents(self, query: str):
        return self.hybrid.get_relevant_documents(query)

