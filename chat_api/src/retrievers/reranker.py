
class RRFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def rerank(self, all_results: list[list[tuple[int, float]]], top_k: int) -> list[tuple[int,float]] :
        rrf_scores = {}

        for results in all_results:
            for rank, (doc_id, _) in enumerate(results):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + self.k + 1)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return sorted_docs[:top_k]


