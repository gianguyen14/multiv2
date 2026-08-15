import numpy as np

from backend.app.config.retrieval_config import RetrievalConfig
from backend.app.indexes.index_factory import IndexFactory
from backend.app.retrieval.pipeline import RetrievalPipeline
from backend.app.retrieval.query_planner import QueryPlanner
from backend.app.retrieval.sharded_retriever import ShardedRetriever


class Encoder:
    def __init__(self, vector):
        self.vector = vector

    def encode_text(self, texts):
        return np.repeat(self.vector[None, :], len(texts), axis=0)


def vectors():
    data = np.eye(4, dtype=np.float32)
    return data, [f"frame_{i}" for i in range(4)]


def test_ivf_training_and_search():
    data, ids = vectors()
    index = IndexFactory.create("ivf", 4, nlist=2)
    index.add(data, ids)
    assert index.search(data[0], 1)[0]["frame_id"] == "frame_0"


def test_sharded_retrieval_and_pipeline():
    data, ids = vectors()
    shards = []
    for part in (slice(0, 2), slice(2, 4)):
        shard = IndexFactory.create("flat", 4)
        shard.add(data[part], ids[part])
        shards.append(shard)
    merged = ShardedRetriever(shards).search(data[0], 2)
    assert merged[0]["frame_id"] == "frame_0"
    pipeline = RetrievalPipeline(Encoder(data[0]), ShardedRetriever(shards))
    assert pipeline.search_text("query", 1)[0]["frame_id"] == "frame_0"


def test_planner_decisions_and_config():
    planner = QueryPlanner(small_dataset=10, large_batch=4)
    assert planner.plan(5, 1).index_type == "flat"
    assert planner.plan(100, 4, recall_critical=True).retriever == "batch"
    assert planner.plan(100, 1, latency_critical=True).retriever == "async"
    assert RetrievalConfig().num_shards == 1
