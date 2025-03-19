from fastapi import FastAPI
from index import IndexDB
from utils import load_configs



settings = load_configs()
data = load_data()

app = FastAPI()
index = IndexDB(settings.embedding.model, )


@app.get("/")
def ask_bot(session_id: int, query: str):
    
    # history = get_history()

    context = index.get_k_results(query, k)

    # response = get_response(history, context, query)

    return {session_id: session_id, text: response}


@app.get("/close_session")
def close_session(session_id: int):
    return {msg: f"Session {session_id} was successfully closed!"}

