import json
from pathlib import Path

import pytest
from PIL import Image

from eval.m13_5_corpus import corpus_statistics, load_corpus


def write_corpus(root, queries=None, candidates=None):
    root.mkdir()
    images = root / "images"
    images.mkdir()
    Image.new("RGB", (2, 2), "red").save(images / "a.png")
    candidates = candidates or [{"candidate_id": "a", "image_path": "images/a.png", "metadata": {"category": "shape"}}]
    queries = queries or [{"query_id": "q", "text": "red", "relevance": {"a": 3}}]
    (root / "candidates.jsonl").write_text("\n".join(json.dumps(row) for row in candidates) + "\n")
    (root / "queries.jsonl").write_text("\n".join(json.dumps(row) for row in queries) + "\n")


def test_valid_corpus_and_statistics(tmp_path):
    root = tmp_path / "corpus"
    write_corpus(root)
    corpus = load_corpus(root)
    report = corpus_statistics(corpus)
    assert report["query_count"] == 1
    assert report["candidate_count"] == 1
    assert len(corpus.fingerprint) == 64


@pytest.mark.parametrize("rows,message", [
    ([{"query_id": "q", "text": "one", "relevance": {"a": 3}}, {"query_id": "q", "text": "two", "relevance": {"a": 2}}], "duplicate query ID"),
    ([{"query_id": "q", "text": "one", "relevance": {"missing": 3}}], "references missing candidate"),
    ([{"query_id": "q", "text": "one", "relevance": {"a": 4}}], "invalid relevance grade"),
    ([{"query_id": "q", "text": "one", "relevance": {"a": 0}}], "no relevant candidate"),
])
def test_invalid_queries(tmp_path, rows, message):
    root = tmp_path / "corpus"
    write_corpus(root, queries=rows)
    with pytest.raises(ValueError, match=message):
        load_corpus(root)


def test_duplicate_candidate_id(tmp_path):
    root = tmp_path / "corpus"
    candidates = [{"candidate_id": "a", "image_path": "images/a.png"}, {"candidate_id": "a", "image_path": "images/a.png"}]
    write_corpus(root, candidates=candidates)
    with pytest.raises(ValueError, match="duplicate candidate ID"):
        load_corpus(root)


def test_missing_image_path(tmp_path):
    root = tmp_path / "corpus"
    write_corpus(root, candidates=[{"candidate_id": "a", "image_path": "images/missing.png"}])
    with pytest.raises(ValueError, match="missing image path"):
        load_corpus(root)
