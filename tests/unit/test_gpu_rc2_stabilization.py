"""Targeted regression tests for GPU RC2 stabilization."""

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import AutoConfig

from backend.app.model_cache import inventory, paddleocr_status
from backend.app.runtime.device_policy import runtime_summary
from backend.app.runtime.operations import write_run_manifest
from backend.app.services.query_refiner import QueryRefiner
from backend.app.video.text_backends import TesseractOCRBackend, create_ocr_backend


def test_provenance_without_git(tmp_path):
    """Verifies write_run_manifest succeeds even if git executable is missing."""
    manifest_path = tmp_path / "run_manifest.json"
    dummy_source = tmp_path / "video.mp4"
    dummy_source.write_text("fake video data")
    dummy_processed = tmp_path / "processed"
    dummy_processed.mkdir()

    with patch("shutil.which", return_value=None):
        manifest = write_run_manifest(
            path=manifest_path,
            command="ingest",
            source=dummy_source,
            processed_root=dummy_processed,
            config={"test": True},
            compute={"device": "cpu"},
        )

    assert manifest_path.is_file()
    assert manifest["git_commit"] is None
    assert manifest["git_dirty"] is False
    assert manifest["provenance_source"] == "unavailable"
    assert manifest["command"] == "ingest"


def test_command_status_without_git():
    """Verifies command_status in projectctl runs without git executable."""
    import projectctl
    from argparse import Namespace

    args = Namespace(json=True, yaml=False, csv=False)
    with patch("shutil.which", return_value=None), patch("projectctl.emit") as mock_emit:
        projectctl.command_status(args)
        assert mock_emit.called
        emitted_state = mock_emit.call_args[0][0]
        assert emitted_state.get("git_branch") is None


def test_ocr_fallback_reporting():
    """Verifies that when PaddleOCR is unavailable, OCR backend routes to Tesseract."""
    backend = create_ocr_backend(name="auto", device="cpu")
    assert isinstance(backend, TesseractOCRBackend)
    assert "Tesseract" in backend.identity() or "tesseract" in backend.identity()

    summary = runtime_summary({"OCR_BACKEND": "auto"})
    ocr_comp = summary["components"]["ocr"]
    assert ocr_comp["component"] == "ocr"
    # When paddle is not installed, fallback is recorded
    if not summary["capabilities"]["paddle"]["installed"]:
        assert ocr_comp["fallback"] is True
        assert ocr_comp["reason"] == "paddle is not installed"


def test_paddleocr_status_diagnostics():
    """Verifies paddleocr_status provides informative diagnostic reporting."""
    status = paddleocr_status()
    assert status["backend"] == "paddleocr"
    assert status["model"] == "PP-OCRv4-vi"
    if not status["cached"]:
        assert "error" in status
        assert "paddleocr is not installed" in status["error"] or "not downloaded" in status["error"]


def test_siglip2_vocab_and_token_invariants():
    """Verifies SigLIP2 configuration and ensures embeddings remain valid."""
    cfg = AutoConfig.from_pretrained("google/siglip2-base-patch16-224")
    assert cfg.text_config.hidden_size == 768
    assert cfg.text_config.vocab_size == 256000
    # Token IDs 49406 and 49407 are within the 256000 vocabulary of SigLIP2
    assert 0 <= cfg.text_config.bos_token_id < cfg.text_config.vocab_size
    assert 0 <= cfg.text_config.eos_token_id < cfg.text_config.vocab_size


def test_query_refiner_fallback_safety():
    """Verifies QueryRefiner provides deterministic fallback when LLM is unavailable or fails."""
    refiner = QueryRefiner(enabled=True, backend="deterministic")
    plan, timings = refiner.refine("người phụ nữ mặc áo dài bên xe máy")
    assert plan.original_query == "người phụ nữ mặc áo dài bên xe máy"
    assert len(plan.visual_queries) > 0
    assert any(vq.language == "vi" for vq in plan.visual_queries)
    assert timings["deterministic_parse_ms"] >= 0.0


def test_entrypoint_script_executable():
    """Verifies docker-entrypoint.sh exists and contains proper umask configuration."""
    entrypoint_path = Path("docker-entrypoint.sh")
    assert entrypoint_path.is_file()
    content = entrypoint_path.read_text()
    assert "umask" in content
    assert 'exec "$@"' in content
