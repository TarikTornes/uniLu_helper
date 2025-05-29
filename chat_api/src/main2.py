from fastapi import FastAPI

from retrievers import DenseRetriever, BM25Retriever, HybridRetriever, RRFusion
from utils import DocStore, log_query

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
bm25_retriever = BM25Retriever([chunks["chunks_dict"][i] for i in sorted(chunks["chunks_dict"])])
reranker = RRFusion()
hybrid_retriever = HybridRetriever([dense_retriever, bm25_retriever], reranker)

model = Gen_Model(settings["generation"]["model"], os.getenv("API_KEY"))
query_opt = QueryModel(settings["generation"]["model"], os.getenv("API_KEY_QUERY"))
chat = ChatDB(host=settings["session"]["host"],
              port=settings["session"]["port"], 
              passwd=os.getenv("PASSWD_REDIS"), 
              dec_resp=True, expiry= settings["session"]["expiry"]*60)


@app.get("/")
def ask_bot(session_id: int, query: str):
    
    history = chat.get_history(session_id)

    opt_query = query_opt.opt_query(query, history)
    
    doc_scores = hybrid_retriever.retrieve(opt_query, settings["retrieval"]["k_nearest"])
    context = []

    for doc_id, score in doc_scores:
        chunk = document_store.get_chunk(doc_id)
        web_link = document_store.get_url(doc_id)
        context.append((doc_id, chunk, web_link, score))

    log_query(context, opt_query)
    log_query(context=context, query=opt_query, overwrite=True)

    response = model.get_response(query_res=context, INSTRUCTION=query, chat_history=history)

    chat.add_message(session_id, query, "user")
    chat.add_message(session_id, response, "assistant")

    return {"session_id": session_id, "text": response}


@app.get("/close_session")
def close_session(session_id: int):
    return {"msg": f"Session {session_id} was successfully closed!"}

