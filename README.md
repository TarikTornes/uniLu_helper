# uniLu_helper
This project contains different approaches of implementing the RAG pipeline:

Method 1:
Dense Retrieval using history and original user query

Method 2:
Dense Retrieval with query rewriting

Method 3:
Query Rewriting -> Hybrid Retrieval (Dense + BM25)

Method 4:
Method 3 + Maximal Margin Relevance Layer

## Prerequisites
- Put embeddings.pkl and chunks.pkl into /data/
- install all dependencies from requirements.txt (see hybridMMR branch)
- install fastapi library
- Configure environment:
  - Create API free API key on groq
  - create .env with `GROQ_API_KEY`

## Run
- got into src folder
- run: `fastapi dev main.py`to start the server
