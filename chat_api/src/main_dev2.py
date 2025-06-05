from src.index import IndexDB
from src.model import Gen_Model, QueryModel, JinaEmbeddings
from src.chats import ChatDB
from src.utils import load_configs, load_data
from src.chains import ConvChain
from dotenv import load_dotenv
from src.utils import DocStore, log_query
from src.retrievers import DenseRetriever, BM25Retriever, HybridRetriever, RRFusion
import os
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

settings = load_configs()
chunks, embeddings = load_data()
load_dotenv()

print(settings["embedding"]["model"])
index = IndexDB(settings["embedding"]["model"], chunks, embeddings)
model = Gen_Model(settings["generation"]["model"], os.getenv("API_KEY"))

document_store = DocStore(chunks["chunks_dict"], chunks["web_page_dict"])
dense_retriever = DenseRetriever(settings["embedding"]["model"], embeddings["embeddings"])
bm25_retriever = BM25Retriever([chunks["chunks_dict"][i] for i in sorted(chunks["chunks_dict"])])
reranker = RRFusion()
hybrid_retriever = HybridRetriever([dense_retriever, bm25_retriever], reranker)

llm = Gen_Model(settings["generation"]["model"], os.getenv("API_KEY"))

os.environ["GROQ_API_KEY"] = os.getenv("API_KEY")


user_chains = {}


def create_chain(session_id: int):
    return ConvChain(llm, hybrid_retriever,document_store)

def get_chain(session_id: int):
    
    if session_id not in user_chains:
        new_chain = create_chain(session_id)
        user_chains[session_id] = new_chain

    return user_chains[session_id]


def ask_bot(session_id:int, query: str):

    chain = get_chain(session_id)

    response = chain.get_response(query, 5)

    return {"session_id": session_id, "text": response}


# Usage example

'''
1.  a.  What do I need to apply for a Master’s?
        -> https://www.uni.lu/en/admissions/bachelor-master/
    b.  Do non-EU students need anything extra?
        Yes—a copy of your European health insurance card, or purchase one upon arrival.

'''

print(ask_bot(200, "What faculties are there at the uni?"))
print(ask_bot(200, "Who are their Deans?"))
