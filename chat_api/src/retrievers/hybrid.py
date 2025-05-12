from .base import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(self, retrievers: list[BaseRetriever], reranker):
        self.retrievers = retrievers
        self.reranker = reranker
    

    def retrieve(self, query: str, k: int) -> list[tuple[int, float]]:

        all_results = [retriever.retrieve(query, k*2) for retriever in self.retrievers]
        return self.reranker.rerank(all_results, k)
