
class DocStore:
    def __init__(self, chunks_dict: dict, web_page_dict: dict):
        self.chunks_dict = chunks_dict
        self.web_page_dict = web_page_dict

    def get_chunk(self, doc_id: int) -> str:
        return self.chunks_dict[doc_id]

    def get_url(self, doc_id: int) -> str:
        return self.web_page_dict[doc_id]

