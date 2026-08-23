"""Unit tests for SigLIP long-query chunk aggregation without model weights."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.app.embeddings.siglip2 import SigLIP2Encoder


class FakeProcessor:
    def __init__(self, token_ids, encoded_texts=None, special_token_count=1):
        self.token_ids = token_ids
        self.tokenizer = FakeTokenizer(encoded_texts or {}, special_token_count)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert len(kwargs["text"]) == len(self.token_ids)
        result = {
            "input_ids": torch.tensor(self.token_ids, dtype=torch.long).reshape(-1, 1),
            "attention_mask": torch.ones((len(self.token_ids), 1), dtype=torch.long),
        }
        return result


class FakeTokenizer:
    def __init__(self, encoded_texts, special_token_count):
        self.encoded_texts = encoded_texts
        self.special_token_count = special_token_count

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return self.encoded_texts.get(text, [0])

    def decode(self, token_ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return "chunk:" + ",".join(str(token_id) for token_id in token_ids)

    def num_special_tokens_to_add(self, pair):
        assert pair is False
        return self.special_token_count


class FakeModel:
    def __init__(self, vectors):
        self.vectors = vectors
        self.seen_token_ids = []

    def get_text_features(self, input_ids, attention_mask):
        del attention_mask
        token_ids = input_ids[:, 0].cpu().tolist()
        self.seen_token_ids.extend(token_ids)
        embeddings = torch.tensor(
            [self.vectors[token_id] for token_id in token_ids], dtype=torch.float32
        )
        return SimpleNamespace(pooler_output=embeddings)


def make_encoder(processor, model, **kwargs):
    encoder = SigLIP2Encoder(device="cpu", revision="test", **kwargs)
    encoder._processor = processor
    encoder._model = model
    encoder._initialized = True
    return encoder


def test_long_text_chunks_are_aggregated_back_to_each_query():
    processor = FakeProcessor(
        token_ids=[1, 2, 3],
        encoded_texts={"a long query": list(range(5)), "short query": [1]},
    )
    model = FakeModel({1: [3.0, 0.0], 2: [0.0, 4.0], 3: [0.0, 5.0]})
    encoder = make_encoder(
        processor,
        model,
        long_text_mode="chunk_mean",
        text_max_length=4,
        text_chunk_stride=1,
    )

    result = encoder.encode_text(["a long query", "short query"], batch_size=2)

    assert result.shape == (2, 2)
    assert np.allclose(result[0], [np.sqrt(0.5), np.sqrt(0.5)])
    assert np.allclose(result[1], [0.0, 1.0])
    assert model.seen_token_ids == [1, 2, 3]
    assert processor.calls[0]["text"] == ["chunk:0,1,2", "chunk:2,3,4", "short query"]
    assert processor.calls[0]["max_length"] == 4


def test_short_text_is_unchanged_by_chunk_mode_when_not_normalized():
    chunk_processor = FakeProcessor(token_ids=[1])
    truncate_processor = FakeProcessor(token_ids=[1])
    chunk_model = FakeModel({1: [3.0, 4.0]})
    truncate_model = FakeModel({1: [3.0, 4.0]})
    chunk_encoder = make_encoder(
        chunk_processor, chunk_model, long_text_mode="chunk_mean"
    )
    truncate_encoder = make_encoder(
        truncate_processor, truncate_model, long_text_mode="truncate"
    )

    chunk_result = chunk_encoder.encode_text("short", normalize=False)
    truncate_result = truncate_encoder.encode_text("short", normalize=False)

    assert np.array_equal(chunk_result, truncate_result)
    assert np.array_equal(chunk_result, np.array([[3.0, 4.0]], dtype=np.float32))
    assert "return_overflowing_tokens" not in truncate_processor.calls[0]


def test_chunk_cap_samples_across_full_query_including_tail(caplog):
    processor = FakeProcessor(
        token_ids=[1, 3, 5],
        encoded_texts={"query with many chunks": list(range(15))},
    )
    model = FakeModel(
        {
            1: [1.0, 0.0],
            2: [2.0, 0.0],
            3: [0.0, 1.0],
            4: [0.0, 2.0],
            5: [1.0, 1.0],
        }
    )
    encoder = make_encoder(
        processor,
        model,
        long_text_mode="chunk_mean",
        text_max_length=4,
        text_chunk_stride=0,
        text_max_chunks=3,
    )

    result = encoder.encode_text("query with many chunks", batch_size=2)

    assert result.shape == (1, 2)
    assert model.seen_token_ids == [1, 3, 5]
    assert np.isclose(np.linalg.norm(result[0]), 1.0)
    assert "sampling across the complete query" in caplog.text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"long_text_mode": "unknown"}, "long_text_mode"),
        ({"text_max_length": 1}, "text_max_length"),
        ({"text_max_length": 8, "text_chunk_stride": 8}, "text_chunk_stride"),
        ({"text_max_chunks": 0}, "text_max_chunks"),
    ],
)
def test_invalid_long_text_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SigLIP2Encoder(device="cpu", revision="test", **kwargs)


def test_processor_without_tokenizer_is_rejected():
    processor = FakeProcessor(token_ids=[1])
    del processor.tokenizer
    model = FakeModel({1: [1.0, 0.0]})
    encoder = make_encoder(processor, model, long_text_mode="chunk_mean")

    with pytest.raises(RuntimeError, match="does not expose a tokenizer"):
        encoder.encode_text("one logical query")
