from .base import BaseRetriever


class HybridRetriever(BaseRetriever):
    def __init__(self, retrievers: list[BaseRetriever], reranker):
        self.retrievers = retrievers
        self.reranker = reranker
    

    def retrieve(self, queries: list[str], k: int = 10) -> list[tuple[int, float]]:

        all_results = []
        for retriever in self.retrievers:
            for query in queries:
                results = retriever.retrieve(query, k * 2)
                all_results.append(results)

        reranked_results = self.reranker.rerank(all_results, 10)


        seen_doc_ids = set()
        unique_results = []

        for doc_id, score in reranked_results:
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_results.append((doc_id, score))
                if len(unique_results) == k:
                    break

        return unique_results
