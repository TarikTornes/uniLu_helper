import os
import json
from dotenv import load_dotenv

# Import necessary components from your main2.py
# Assuming these are available in the same directory or accessible via PYTHONPATH
from retrievers import DenseRetriever, BM25Retriever, HybridRetriever, RRFusion
from utils import DocStore, load_configs, load_data # Assuming load_configs and load_data are in utils
from chats import ChatDB # Needed for chat history for multi-turn queries
from model import QueryModel # Needed for query optimization in multi-turn queries

# Import the benchmark utility
from benchmark_utils import RetrievalBenchmark

# Load environment variables
load_dotenv()

# --- Initialize RAG Components (similar to your main2.py) ---
settings = load_configs()
chunks, embeddings = load_data()

document_store = DocStore(chunks["chunks_dict"], chunks["web_page_dict"])
dense_retriever = DenseRetriever(settings["embedding"]["model"], embeddings["embeddings"])
dense_retr_mmr = DenseRetriever(settings["embedding"]["model"], embeddings["embeddings"])
bm25_retriever = BM25Retriever([chunks["chunks_dict"][i] for i in sorted(chunks["chunks_dict"])])
reranker = RRFusion(35)
hybrid_retriever = HybridRetriever([dense_retriever, bm25_retriever], reranker, dense_retr_mmr)

# For multi-turn queries, we need the query optimization model
query_opt = QueryModel(settings["generation"]["model"], os.getenv("API_KEY_QUERY"))
chat_db = ChatDB(host=settings["session"]["host"],
                 port=settings["session"]["port"],
                 passwd=os.getenv("PASSWD_REDIS"),
                 dec_resp=True, expiry= settings["session"]["expiry"]*60)

# --- Define the retrieval function for the benchmark ---
def benchmark_retrieval_function(query_text: str, chat_history: list) -> list:
    """
    This function simulates the retrieval process of your RAG system.
    It takes a query and chat history, performs query optimization (if applicable),
    and then calls the hybrid retriever.

    Args:
        query_text (str): The current query text.
        chat_history (list): A list of dictionaries representing previous turns in the conversation.
                             Each dict should have 'role' and 'parts' (e.g., [{"role": "user", "parts": [{"text": "..."}]}]).

    Returns:
        list: A list of tuples, where each tuple is (chunk_id, score).
              Example: [(123, 0.9), (456, 0.8), ...]
    """
    print(f"  --> Optimizing query for retrieval: '{query_text}' with history: {chat_history}")
    # Perform query optimization using the loaded model
    optimized_queries = query_opt.opt_query(query_text, chat_history)
    print(f"  --> Optimized queries: {optimized_queries}")

    # Retrieve documents using your hybrid retriever
    # The k_nearest parameter here is for the retriever, not the benchmark's Recall@k.
    # It should be large enough to potentially cover the relevant chunks.
    retrieved_doc_scores = hybrid_retriever.retrieve(optimized_queries, settings["retrieval"]["k_nearest"])
    print(f"  --> Retrieved {len(retrieved_doc_scores)} chunks.")
    return retrieved_doc_scores

# --- Run the Benchmark ---
if __name__ == "__main__":
    queries_manifest_path = 'queries_manifest.json'
    gold_equivalence_path = 'gold_equivalence.json'

    benchmark = RetrievalBenchmark(queries_manifest_path, gold_equivalence_path)

    # Define the K for Recall@K evaluation
    # This determines how many top retrieved chunks are considered a "hit"
    RECALL_K = 5 # You can adjust this value

    print(f"Running benchmark with K={RECALL_K}...")
    overall_results = benchmark.run_full_benchmark(benchmark_retrieval_function, RECALL_K)

    print("\n--- Detailed Benchmark Results ---")
    for query_id, result in overall_results['detailed_results'].items():
        print(f"\nQuery ID: {query_id}")
        print(f"  Query Text: {benchmark.get_queries()[query_id]['query_text']}")
        print(f"  Recall@{RECALL_K} Hit: {result['recall_at_k']}")
        print(f"  Number of Gold Equivalence Sets: {result['num_gold_sets']}")
        print(f"  Relevant Sets Hit: {result['relevant_sets_hit']}")
        print(f"  Retrieved Chunk IDs (Top {RECALL_K}): {result['retrieved_ids']}")
        # Optionally, you can print which gold chunks were expected vs retrieved
        gold_sets = benchmark.get_gold_chunks(query_id)
        print(f"  Expected Gold Sets: {gold_sets}")
        print("-" * 30)

    print("\nBenchmark execution complete.")

