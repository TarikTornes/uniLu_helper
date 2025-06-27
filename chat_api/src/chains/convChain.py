from utils import log_query

class ConvChain:

    def __init__(self, model, retriever, doc_store):
        self.model = model
        self.retriever = retriever
        self.history = []
        self.document_store = doc_store
        self.retrieved_ids = []

    def get_response(self, query, k):
        hist = self.get_hist(5)
        hist_str = self.to_string(hist)
        scores = self.retriever.retrieve(hist_str + query, k)
        context = self.get_chunks(scores, query)
        resp = self.model.get_response(context, query, hist)
        self.history.append({"role":"user", "content":query})
        self.history.append({"role":"assistant", "content":resp})
        return resp


    def get_hist(self, r=5):
        l = len(self.history)
        return self.history[-(min(2*r,l)):]

    def to_string(self, hist):
        res = ""
        for msg in hist:
            res = res + msg["content"]

        return res

    def get_chunks(self, doc_scores, query):

        context = []

        for doc_id, score in doc_scores:
            chunk = self.document_store.get_chunk(doc_id)
            web_link = self.document_store.get_url(doc_id)
            context.append((doc_id, chunk, web_link, score))
            self.retrieved_ids.append(doc_id)

        log_query(context, [query], query)
        log_query(context=context, 
                  opt_queries=[query], 
                  og_query=query, 
                  overwrite=True)


        return context


