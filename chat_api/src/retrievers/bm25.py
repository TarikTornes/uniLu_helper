from rank_bm25 import BM25Okapi
from .base import BaseRetriever
import numpy as np

class BM25Retriever(BaseRetriever):
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)


    def _tokenize(self, text:str) -> list[str]:
        return text.lower().split()

    def retrieve(self, query: str, k: int) -> list[tuple[int, float]]:
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:k]
        return [(int(i), float(doc_scores[i])) for i in top_indices]
