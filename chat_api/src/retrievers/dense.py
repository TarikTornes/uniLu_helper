import faiss, re
import numpy as np
from transformers import AutoModel
from .base import BaseRetriever

class DenseRetriever(BaseRetriever):
    def __init__(self, embeddings_path:str, embeddings):
        self.embeddings_model = AutoModel.from_pretrained(embeddings_path, trust_remote_code=True, device_map="auto")
        self.embeddings = embeddings.astype('float32')
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def encode_query(self, query:str) -> np.ndarray:
        query_embedding = self.embeddings_model.encode([query])
        faiss.normalize_L2(query_embedding)
        return query_embedding

    def retrieve(self, query: str, k: int) -> list[tuple[int,float]]:

        query_embedding = self.encode_query(query)

        scores, indices = self.index.search(query_embedding, k)

        # returns a list of tuples [(idx1, score1), (idx2,score2), ...]
        return list(zip(indices[0].tolist(), scores[0].tolist()))

