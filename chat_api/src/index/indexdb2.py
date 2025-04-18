import utils
from utils import log
from utils import check_device

from transformers import AutoConfig, AutoModel
import faiss, re
#from sklearn.preprocessing import normalize


class IndexDB2:

    def __init__(self, embeddings_path, chunks_dict, embeddings):

        check_device()
        self.embeddings_config = AutoConfig.from_pretrained(embeddings_path)
        self.embeddings_model = AutoModel.from_pretrained(embeddings_path, trust_remote_code=True, device_map="auto")
        check_device()
        self.embeddings = embeddings["embeddings"]
        self.chunks_dict = chunks_dict["chunks_dict"]
        self.web_page_dict = chunks_dict["web_page_dict"]

        self.hidden_size = self.embeddings_config.hidden_size
        print(self.hidden_size)

        # Add normalization step
        # self.embeddings = normalize(self.embeddings, axis=1)
        faiss.normalize_L2(self.embeddings)

        self.index = faiss.IndexFlatIP(self.hidden_size)
        
        self.index.add(self.embeddings)

        log("INFO", "Query: EmbeddingDB successfully loaded")
        check_device()



    def get_k_results(self, QUERY, k):
        results = []

        query = self.embeddings_model.encode([QUERY])
        # Added normalization
        #query = normalize(self.embeddings_model.encode([QUERY]))
        faiss.normalize_L2(query)

        D, I = self.index.search(query, k)

        log("QUERY", None, QUERY)

        for i, j in enumerate(I[0]):
            vec = self.embeddings[j]
            chunk = self.chunks_dict[j]
            web_link = self.web_page_dict[j]
            results.append((j, chunk, web_link))
            mess = f'Top: {i}, Chunk: {j}\nSimilarity:  {round(D[0][i], 3)}\n\n {self.chunks_dict[j]}'
            log("QUERY_RESULTS", mess + f"\n URL: {web_link}")

        return results




