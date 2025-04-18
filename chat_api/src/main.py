from fastapi import FastAPI
from index import IndexDB2
from chats import ChatDB
from model import Gen_Model, QueryModel
from utils import load_configs, load_data
from dotenv import load_dotenv
import os



settings = load_configs()
chunks, embeddings = load_data()
load_dotenv()

app = FastAPI()
index = IndexDB2(settings["embedding"]["model"], chunks, embeddings)
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

    context = index.get_k_results(opt_query, settings["retrieval"]["k_nearest"])

    response = model.get_response(query_res=context, INSTRUCTION=query, chat_history=history)

    chat.add_message(session_id, query, "user")
    chat.add_message(session_id, response, "assistant")

    return {"session_id": session_id, "text": response}


@app.get("/close_session")
def close_session(session_id: int):
    return {"msg": f"Session {session_id} was successfully closed!"}

