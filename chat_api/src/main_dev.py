from index import IndexDB
from utils import load_configs, load_data



settings = load_configs()
chunks, embeddings = load_data()

print(settings.embedding.model)
index = IndexDB(settings.embedding.model, chunks, embeddings)


def ask_bot(session_id: int, query: str):
    
    #history = get_history()

    context = index.get_k_results(query, 10)

    # response = get_response(history, context, query)

    return {"session_id": session_id, "text": context}


def close_session(session_id: int):
    return {"msg": f"Session {session_id} was successfully closed!"}


ask_bot(200, "Who is the director of the University of Luxembourg")
