from typing import List
from langchain_core.embeddings import Embeddings  
import numpy as np

class JinaEmbeddings(Embeddings):
    def __init__(self, model):
        """ model: your JinaAI AutoModel that has a .encode(list[str])->np.ndarray """
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # encode returns a 2D numpy array shape (len(texts), dim)
        embs: np.ndarray = self.model.encode(texts)
        # ensure float and list-of-lists format
        return embs.astype("float32").tolist()

    def embed_query(self, text: str) -> List[float]:
        # encode single query, returns array shape (1, dim)
        q_emb: np.ndarray = self.model.encode([text])
        return q_emb.astype("float32")[0].tolist()
