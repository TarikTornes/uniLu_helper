import json
from pathlib import Path
from . import log_query


class Benchmark_UniBot:

    def __init__(self,
                 confs,
                 query_optimizer,
                 gen_model,
                 chat_db,
                 doc_store,
                 retriever):
        self.settings = confs
        self.query_opt = query_optimizer
        self.model = gen_model
        self.chat = chat_db
        self.document_store = doc_store
        self.retriever = retriever



    def load_gold(self, path_json):
        json_path = Path(path_json)
        with json_path.open(encoding="utf-8") as f:
            self.gold = json.load(f)

    def load_manifest(self, path_json):
        json_path = Path(path_json)
        with json_path.open(encoding="utf-8") as f:
            self.manifest = json.load(f)

    def load_tests():
        """ This function loads the configurations for every evaluation run."""
        pass


    def compute_set_precision_recall(self, retrieved_ids, gold_equivalence_sets, K):
        """
        retrieved_ids: top-K chunk IDs, in descending score order
        gold_equivalence_sets: e.g. [[20,21,22,23,24,25], [30,31], [40]]
        """
        topk = retrieved_ids
        print("LENGTH:   ",len(topk))
        print(gold_equivalence_sets)
        print(retrieved_ids)
        covered_sets = 0

        for eq_set in gold_equivalence_sets:
            # “Hit” if ANY member of eq_set appears in top-K
            for chunk_id in eq_set:
                if chunk_id in retrieved_ids:
                    covered_sets += 1
                    break

        print(covered_sets)
        precision_at_K = covered_sets / K
        recall_at_K    = covered_sets / len(gold_equivalence_sets)
        return precision_at_K, recall_at_K


    def run_benchmark(self,k):

        res_dict = {"prec":[], "recall": []}

        for q_id in self.manifest:
            res_dict = self.perform_test(99999, q_id, res_dict, k)
            
            print(f'Finished with query {q_id} :    Precision: {res_dict["prec"]}   Recall: {res_dict["recall"]}')

        return res_dict




    def perform_test(self, session_id: int, query_id: int, res_dict: dict,k:int, follow_up:bool = False):
        retrieved_ids = []

        query_obj = self.manifest[query_id]

        if follow_up == False:
            self.chat.kill_session_history(session_id)
            history = self.chat.get_history(session_id)
            og_query = query_obj["query_text"]
            opt_queries = self.query_opt.opt_query(og_query, history)
        else:
            history = self.chat.get_history(session_id)
            og_query = query_obj["query_followup"]
            opt_queries = self.query_opt.opt_query(og_query,history)

        doc_scores = self.retriever.retrieve(opt_queries, k)

        context = []


        for doc_id, score in doc_scores:
            chunk = self.document_store.get_chunk(doc_id)
            web_link = self.document_store.get_url(doc_id)
            context.append((doc_id, chunk, web_link, score))
            retrieved_ids.append(doc_id)

        log_query(context, opt_queries, og_query)
        log_query(context=context, 
                  opt_queries=opt_queries, 
                  og_query=og_query, 
                  overwrite=True)

        response = self.model.get_response(query_res=context, INSTRUCTION=og_query, chat_history=history)

        self.chat.add_message(session_id, og_query, "user")
        self.chat.add_message(session_id, response, "assistant")

        if follow_up:
            equivalence_sets = self.gold[query_id + "_followup"]
        else:
            equivalence_sets = self.gold[query_id]

        res_prec, res_recall = self.compute_set_precision_recall(retrieved_ids,
                                                                 equivalence_sets["gold_equivalence_sets"],
                                                                 k)
        
        res_dict["prec"].append(res_prec)
        res_dict["recall"].append(res_recall)


        if query_obj["type"] == "multi" and follow_up == False:
            res_dict = self.perform_test(session_id,query_id, res_dict, k, True)


        return res_dict






