import json
from pathlib import Path
import pytest

from tools.benchmark_retrieval import (
    BenchmarkQueryInput,
    ConcentrationMetrics,
    ProductionBenchmarkRunner,
    compute_concentration,
    compute_hhi,
    compute_latency_stats,
    compute_rank_metrics,
    localize_failure,
    simulate_diversity_soft_cap,
    simulate_round_robin_diversification,
)


def test_hhi_calculation():
    assert compute_hhi([]) == 0.0
    # Single dominant video (100% share) -> HHI = 1.0
    assert compute_hhi([1.0]) == 1.0
    # Equal distribution among 4 videos (0.25 each) -> HHI = 4 * 0.0625 = 0.25
    assert pytest.approx(compute_hhi([0.25, 0.25, 0.25, 0.25]), 0.001) == 0.25


def test_concentration_empty_and_normal():
    # Empty candidates
    empty_res = compute_concentration([], depth=100)
    assert empty_res.total_candidates == 0
    assert empty_res.unique_videos == 0
    assert empty_res.hhi == 0.0

    # Normal candidates
    candidates = [
        {"video_id": "V1", "frame_id": 10},
        {"video_id": "V1", "frame_id": 20},
        {"video_id": "V1", "frame_id": 30},
        {"video_id": "V2", "frame_id": 40},
        {"video_id": "V3", "frame_id": 50},
    ]
    res = compute_concentration(candidates, depth=5)
    assert res.total_candidates == 5
    assert res.unique_videos == 3
    assert res.top_video_id == "V1"
    assert res.top_video_count == 3
    assert res.top_video_share == 0.6
    assert res.top_3_video_share == 1.0
    assert pytest.approx(res.hhi, 0.001) == (0.6**2 + 0.2**2 + 0.2**2)


def test_rank_metrics_calculations():
    # Empty ranks
    empty_m = compute_rank_metrics([])
    assert empty_m["total_queries"] == 0
    assert empty_m["mrr"] == 0.0

    # 4 queries: ranks [1, 5, 20, None]
    ranks = [1, 5, 20, None]
    m = compute_rank_metrics(ranks)
    assert m["total_queries"] == 4
    assert m["evaluated_queries"] == 3
    assert m["hits"]["Hit@1"] == 0.25
    assert m["hits"]["Hit@5"] == 0.50
    assert m["hits"]["Hit@10"] == 0.50
    assert m["hits"]["Hit@20"] == 0.75
    assert m["hits"]["Hit@50"] == 0.75
    assert m["hits"]["Hit@100"] == 0.75
    # MRR = (1/1 + 1/5 + 1/20 + 0) / 4 = (1 + 0.2 + 0.05) / 4 = 1.25 / 4 = 0.3125
    assert pytest.approx(m["mrr"], 0.001) == 0.3125
    assert m["median_rank"] == 5


def test_latency_stats_calculations():
    assert compute_latency_stats([])["mean_ms"] == 0.0
    stats = compute_latency_stats([100.0, 200.0, 300.0, 400.0, 500.0])
    assert stats["min_ms"] == 100.0
    assert stats["max_ms"] == 500.0
    assert stats["mean_ms"] == 300.0
    assert stats["p50_ms"] == 300.0


def test_query_input_loading_json_and_jsonl(tmp_path):
    # Test JSON list loading
    json_data = [
        {"id": "Q1", "type": "kis", "query": "xe cuu hoa", "video_id": "V1", "accepted_frame_interval": [10, 50]},
        {"id": "Q2", "type": "qa", "query": "bien so gi?", "ground_truth": {"video_id": "V2"}},
    ]
    json_file = tmp_path / "queries.json"
    json_file.write_text(json.dumps(json_data), encoding="utf-8")

    queries = ProductionBenchmarkRunner.load_queries(json_file)
    assert len(queries) == 2
    assert queries[0].query_id == "Q1"
    assert queries[0].task_type == "kis"
    assert queries[0].ground_truth_video_ids == ("V1",)
    assert queries[0].accepted_frame_intervals == ((10, 50),)
    assert queries[1].query_id == "Q2"
    assert queries[1].task_type == "qa"
    assert queries[1].ground_truth_video_ids == ("V2",)

    # Test JSONL loading
    jsonl_file = tmp_path / "queries.jsonl"
    jsonl_file.write_text(
        '{"id": "Q3", "task": "kis", "text": "thuyen tren song"}\n{"id": "Q4", "type": "trake", "events": ["a", "b"]}\n',
        encoding="utf-8",
    )
    queries_jl = ProductionBenchmarkRunner.load_queries(jsonl_file)
    assert len(queries_jl) == 2
    assert queries_jl[0].query_id == "Q3"
    assert queries_jl[0].query_text == "thuyen tren song"
    assert queries_jl[1].query_id == "Q4"
    assert queries_jl[1].task_type == "trake"
    assert queries_jl[1].query_text == "a -> b"


def test_frozen_query_input_copies_mutable_collections():
    video_ids = ["V1"]
    intervals = [[10, 50]]
    query = BenchmarkQueryInput(
        "Q1",
        "kis",
        "query",
        events=["event"],
        ground_truth_video_ids=video_ids,
        accepted_frame_intervals=intervals,
    )

    video_ids.append("V2")
    intervals[0][0] = 0

    assert query.events == ("event",)
    assert query.ground_truth_video_ids == ("V1",)
    assert query.accepted_frame_intervals == ((10, 50),)


def test_failure_localization_stages():
    # Success case
    cls_name, ev, stage = localize_failure(
        gt_video_ids=["V1"],
        gt_intervals=[],
        plan=None,
        channel_hits={},
        fused_results=[],
        reranked_results=[],
        final_results=[],
        target_rank=3,
        visual_concentration=None,
    )
    assert cls_name == "SUCCESS"

    # No GT
    cls_name, ev, stage = localize_failure(
        gt_video_ids=[],
        gt_intervals=[],
        plan=None,
        channel_hits={},
        fused_results=[],
        reranked_results=[],
        final_results=[],
        target_rank=None,
        visual_concentration=None,
    )
    assert cls_name == "UNKNOWN"

    # Visual recall failure
    cls_name, ev, stage = localize_failure(
        gt_video_ids=["V_MISSING"],
        gt_intervals=[],
        plan=None,
        channel_hits={"visual": {("V1", 10): 0.5}},
        fused_results=[{"video_id": "V1"}],
        reranked_results=[{"video_id": "V1"}],
        final_results=[{"video_id": "V1"}],
        target_rank=None,
        visual_concentration=ConcentrationMetrics(
            depth=10, total_candidates=10, unique_videos=5, top_video_id="V1",
            top_video_count=2, top_video_share=0.2, top_3_video_share=0.5,
            top_5_video_share=0.8, hhi=0.1
        ),
    )
    assert cls_name == "VISUAL_RECALL_FAILURE"
    assert stage == "visual_retrieval"

    # Concentration failure
    cls_name, ev, stage = localize_failure(
        gt_video_ids=["V_MISSING"],
        gt_intervals=[],
        plan=None,
        channel_hits={"visual": {("V1", 10): 0.5}},
        fused_results=[{"video_id": "V1"}],
        reranked_results=[{"video_id": "V1"}],
        final_results=[{"video_id": "V1"}],
        target_rank=None,
        visual_concentration=ConcentrationMetrics(
            depth=100, total_candidates=100, unique_videos=2, top_video_id="V1",
            top_video_count=85, top_video_share=0.85, top_3_video_share=1.0,
            top_5_video_share=1.0, hhi=0.75
        ),
    )
    assert cls_name == "CANDIDATE_CONCENTRATION"


def test_diversity_simulations():
    candidates = [
        {"video_id": "V1", "frame_id": 1, "score": 10},
        {"video_id": "V1", "frame_id": 2, "score": 9},
        {"video_id": "V1", "frame_id": 3, "score": 8},
        {"video_id": "V1", "frame_id": 4, "score": 7},
        {"video_id": "V2", "frame_id": 5, "score": 6},
        {"video_id": "V3", "frame_id": 6, "score": 5},
    ]

    # Soft cap: max 2 per video
    capped = simulate_diversity_soft_cap(candidates, max_per_video=2, top_k=4)
    assert len(capped) == 4
    capped_vids = [c["video_id"] for c in capped]
    assert capped_vids == ["V1", "V1", "V2", "V3"]

    # Round robin
    rr = simulate_round_robin_diversification(candidates, top_k=4)
    assert len(rr) == 4
    rr_vids = [c["video_id"] for c in rr]
    assert rr_vids == ["V1", "V2", "V3", "V1"]


def test_unlabeled_queries_do_not_produce_fake_rank_metrics():
    queries = [
        BenchmarkQueryInput("labeled", "kis", "query", ground_truth_video_ids=("V1",)),
        BenchmarkQueryInput("unlabeled", "kis", "diagnostic only"),
    ]
    runner = ProductionBenchmarkRunner(
        lambda query, top_k: [{"video_id": "V2", "frame_id": 1}]
    )

    report = runner.run(queries, top_k=1)

    assert report["rank_metrics"]["total_queries"] == 1
    assert report["rank_metrics"]["mrr"] == 0.0
    assert report["quality_metrics_status"] == "available"
    assert report["unlabeled_queries"] == 1


def test_all_unlabeled_queries_report_quality_metrics_unavailable():
    runner = ProductionBenchmarkRunner(
        lambda query, top_k: [{"video_id": "V1", "frame_id": 1}]
    )

    report = runner.run(
        [BenchmarkQueryInput("diagnostic", "kis", "query without labels")],
        top_k=1,
    )

    assert report["rank_metrics"] is None
    assert report["quality_metrics_status"] == "unavailable_no_ground_truth"
    assert report["unlabeled_queries"] == 1


def test_concentration_rejects_candidates_without_video_identity():
    with pytest.raises(ValueError, match="must have a video_id"):
        compute_concentration([{"frame_id": 1}])


def test_query_loader_rejects_ranges_without_video_ground_truth(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            [{"id": "Q1", "query": "query", "accepted_frame_interval": [1, 2]}]
        )
    )
    with pytest.raises(ValueError, match="require a ground-truth video ID"):
        ProductionBenchmarkRunner.load_queries(path)
