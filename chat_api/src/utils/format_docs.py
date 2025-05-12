
def format_documents(documents, max_docs=None):
    if max_docs:
        documents = documents[:max_docs]

    formatted = ""

    for rank, (doc_id, text, url, similarity) in enumerate(documents, start=1):

        formatted += f"Rank: {rank}\n"
        formatted += f"Document ID: {doc_id}\n"
        formatted += f"Text: {text}\n"
        formatted += f"URL: {url}\n"
        formatted += f"Similarity Score: {similarity:.4f}\n\n"

    return formatted.strip()
