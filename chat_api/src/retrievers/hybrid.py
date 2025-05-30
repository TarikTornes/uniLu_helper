from .base import BaseRetriever
import numpy as np


class HybridRetriever(BaseRetriever):
    def __init__(self, retrievers: list[BaseRetriever], 
                 reranker, 
                 dense_retr,
                 lambda_mmr: float=0.7):
        self.retrievers = retrievers
        self.reranker = reranker
        self.dense = dense_retr
        self.lambda_mmr = lambda_mmr
    

    def retrieve(self, queries: list[str], k: int = 10) -> list[tuple[int, float]]:

        all_results = []
        for retriever in self.retrievers:
            for query in queries:
                results = retriever.retrieve(query, k * 2)
                all_results.append(results)

        reranked = self.reranker.rerank(all_results, min(k * 5, k*2))

        seen_doc_ids = set()
        unique_res = []

        for doc_id, score in reranked:
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_res.append((doc_id, score))

        candidates_ids = [doc_id for doc_id, _ in unique_res]
        scores = np.array([score for _, score in unique_res], dtype=np.float32)

        embeddings = self.dense.embeddings[candidates_ids]

        sim_matrix = embeddings @ embeddings.T

        selected = []


        most_relevant_idx = int(np.argmax(scores))

        selected.append(most_relevant_idx)


        while len(selected) < k:

            unselected = [i for i in range(len(candidates_ids)) if i not in selected]
            mmr_vals = []

            for i in unselected:
                relevance = scores[i]
                diversity = max(sim_matrix[i][j] for j in selected)
                mmr_score = (self.lambda_mmr * relevance - (1-self.lambda_mmr) * diversity)
                mmr_vals.append((i, mmr_score))

            next_idx, _ = max(mmr_vals, key = lambda x: x[1])
            selected.append(next_idx)

        return [(candidates_ids[i], scores[i]) for i in selected]







