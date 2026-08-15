from concurrent.futures import ThreadPoolExecutor


class ThreadRPC:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def call(self, worker, query, top_k):
        return self.executor.submit(worker.search, query, top_k)
