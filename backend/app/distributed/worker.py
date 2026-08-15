class FaissWorker:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def search(self, query, top_k):
        if hasattr(self.pipeline, "search"):
            return self.pipeline.search(query, top_k)
        return self.pipeline.search_text(query, top_k)
