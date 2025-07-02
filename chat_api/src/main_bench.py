from fastapi import FastAPI

from retrievers import DenseRetriever, BM25Retriever, HybridRetriever, RRFusion
from utils import DocStore, log_query, Benchmark_UniBot


from chats import ChatDB
from model import Gen_Model, QueryModel
from utils import load_configs, load_data
from dotenv import load_dotenv
import os


settings = load_configs()
chunks, embeddings = load_data()
load_dotenv()

app = FastAPI()

document_store = DocStore(chunks["chunks_dict"], chunks["web_page_dict"])
dense_retriever = DenseRetriever(settings["embedding"]["model"], embeddings["embeddings"])
dense_retr_mmr = DenseRetriever(settings["embedding"]["model"], embeddings["embeddings"])
bm25_retriever = BM25Retriever([chunks["chunks_dict"][i] for i in sorted(chunks["chunks_dict"])])
reranker = RRFusion(35)
hybrid_retriever = HybridRetriever([dense_retriever, bm25_retriever], reranker, dense_retr_mmr)

model = Gen_Model(settings["generation"]["model"], os.getenv("API_KEY"))
query_opt = QueryModel(settings["generation"]["model"], os.getenv("API_KEY_QUERY"))
chat = ChatDB(host=settings["session"]["host"],
              port=settings["session"]["port"], 
              passwd=os.getenv("PASSWD_REDIS"), 
              dec_resp=True, expiry= settings["session"]["expiry"]*60)

manifest = None
quers= {}


def run_bench():

    bench = Benchmark_UniBot(settings,
                            query_opt,
                            model,
                            chat,
                            document_store,
                            hybrid_retriever)

    bench.load_gold("../data/benchmark/gold_equivalence.json")
    bench.load_manifest("../data/benchmark/queries_manifest.json")

    ks = [1,3,5,10,15,20]
    results_at_k = {}

    for k in ks:
        res_dict = {}
        res_dict = bench.run_benchmark(k)

        mean_precision = sum(qres for qres in res_dict["prec"]) / len(res_dict["prec"])
        mean_recall= sum(qres for qres in res_dict["recall"]) / len(res_dict["recall"])

        print(f'-----------------   K = {k}  -------------------------')
        print("Precision:  ", mean_precision)
        print("Recall:  ", mean_recall)
        print(f'------------------------------------------------------')

        results_at_k[k] = (mean_precision,mean_recall)

    print()
    print()
    print(results_at_k)





run_bench()
