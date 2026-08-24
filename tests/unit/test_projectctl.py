import json
import os
from argparse import Namespace
import subprocess
import sys

import pytest

import projectctl


def args(**values):
    return Namespace(json=True, verbose=False, output=None, **values)


def test_env_json_reports_runtime(capsys):
    assert projectctl.main(["env", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["python"]["executable"]
    assert "packages" in result and "compute" in result


def test_doctor_reports_optional_blockers(capsys, monkeypatch):
    monkeypatch.setattr(projectctl, "executable", lambda name: None)
    monkeypatch.setattr(projectctl, "_module", lambda name: name == "faster_whisper")
    monkeypatch.setattr(projectctl, "model_inventory", lambda model=None: {
        "visual": {"cached": False}, "asr": {"cached": False}})
    assert projectctl.main(["doctor", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ocr_executable"] == "MISSING"
    assert result["ocr_english"] == "MISSING" and result["ocr_vietnamese"] == "MISSING"
    assert result["browser_e2e"] == "OPTIONAL DEPENDENCY MISSING"


def test_search_json_and_export(monkeypatch, tmp_path, capsys):
    rows = [{"video_id": "v", "frame_id": 4, "score": 1.0}]
    monkeypatch.setattr(projectctl, "search_rows", lambda query, top_k=100: rows)
    output = tmp_path / "results.json"
    assert projectctl.main(["search", "query", "--json", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == rows
    assert json.loads(output.read_text()) == rows


def test_missing_ocr_fails_actionably(monkeypatch, capsys):
    monkeypatch.setattr(projectctl, "executable", lambda name: None)
    assert projectctl.main(["ocr", "video.mp4"]) == 1
    assert "install tesseract" in capsys.readouterr().err


def test_competition_evaluate_missing_data_is_clear(tmp_path, capsys):
    assert projectctl.main(["evaluate", "--competition", "--ground-truth", str(tmp_path)]) == 1
    assert "missing competition ground truth" in capsys.readouterr().err


def test_dataset_report_does_not_require_ground_truth(tmp_path, capsys):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    assert projectctl.main(["dataset", "report", str(tmp_path), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["video_count"] == 1 and result["source_bytes"] == 5
    assert "ground_truth" not in result


def test_preprocess_preflight_only_stops_before_all_stages(monkeypatch, tmp_path, capsys):

    processed = tmp_path / "processed"
    calls = []
    monkeypatch.setattr(projectctl, "ingest_report", lambda args: calls.append("preflight") or {
        "preflight": {"ok": True, "errors": []}})
    monkeypatch.setattr(projectctl, "_text_pipeline", lambda *args, **kwargs: calls.append("text"))
    monkeypatch.setattr(projectctl, "executable", lambda name: calls.append(f"executable:{name}") or True)
    monkeypatch.setattr(projectctl, "_module", lambda name: calls.append(f"module:{name}") or True)
    assert projectctl.main(["preprocess", str(tmp_path), "--processed-root", str(processed),
        "--preflight-only", "--json"]) == 0
    assert calls == ["preflight"]
    assert not processed.exists()
    assert json.loads(capsys.readouterr().out)["preflight"]["ok"] is True


def test_preflight_failure_is_reported_and_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(projectctl, "ingest_report", lambda args: {
        "preflight": {"ok": False, "errors": ["source path is missing"]}})
    assert projectctl.main(["ingest", "missing", "--preflight-only", "--json"]) == 1
    output = capsys.readouterr()
    assert json.loads(output.out)["preflight"]["ok"] is False
    assert "source path is missing" in output.err


def test_normal_preprocess_preserves_stage_order(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(projectctl, "ingest_report", lambda args: calls.append("ingest") or {"videos_succeeded": 1})
    monkeypatch.setattr(projectctl, "executable", lambda name: True)
    monkeypatch.setattr(projectctl, "_module", lambda name: True)
    monkeypatch.setattr(projectctl, "_require_models", lambda **kwargs: None)
    monkeypatch.setattr(projectctl, "_text_pipeline", lambda args, use_ocr, use_asr:
        calls.append("ocr" if use_ocr else "asr") or {"videos_succeeded": 1})
    assert projectctl.main(["preprocess", "videos", "--json"]) == 0
    assert calls == ["ingest", "ocr", "asr"]
    capsys.readouterr()


def test_preprocess_keeps_optional_modalities_fail_open(monkeypatch, capsys):
    monkeypatch.setattr(projectctl, "ingest_report", lambda args: {"videos_succeeded": 1})
    monkeypatch.setattr(projectctl, "executable", lambda name: name == "tesseract")
    monkeypatch.setattr(projectctl, "_module", lambda name: True)
    monkeypatch.setattr(projectctl, "_text_pipeline", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")))
    monkeypatch.setattr(projectctl, "_require_models", lambda **kwargs: None)
    assert projectctl.main(["preprocess", "videos", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ingest"]["videos_succeeded"] == 1
    assert result["ocr"].startswith("unavailable") and result["asr"].startswith("unavailable")


def inventory(visual=False, asr=False):
    return {
        "visual": {"backend": "siglip2", "model": "google/siglip2-base-patch16-224", "cached": visual, "cache_path": None},
        "asr": {"backend": "faster-whisper", "model": "small", "cached": asr, "cache_path": None},
        "ocr": {"backend": "tesseract", "eng": True, "vie": True},
    }


def test_models_help(capsys):
    try:
        projectctl.parser().parse_args(["models", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "--verify-offline" in capsys.readouterr().out


def test_models_inventory_json(monkeypatch, capsys):
    monkeypatch.setattr(projectctl, "model_inventory", lambda model=None: inventory(True, False))
    assert projectctl.main(["models", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["visual"]["model"] == "google/siglip2-base-patch16-224"
    assert result["asr"]["model"] == "small"


def test_models_selective_prepare(monkeypatch, capsys):
    states = iter((inventory(False, False), inventory(True, False)))
    monkeypatch.setattr(projectctl, "model_inventory", lambda model=None: next(states))
    import backend.app.model_cache as cache
    calls = []
    monkeypatch.setattr(cache, "prepare_visual", lambda: calls.append("visual"))
    monkeypatch.setattr(cache, "prepare_asr", lambda model: calls.append("asr"))
    assert projectctl.main(["models", "--prepare", "--visual", "--json"]) == 0
    assert calls == ["visual"]
    assert set(json.loads(capsys.readouterr().out)) == {"visual", "ocr"}


def test_models_dry_run_does_not_prepare(monkeypatch, capsys):
    monkeypatch.setattr(projectctl, "model_inventory", lambda model=None: inventory(False, False))
    assert projectctl.main(["models", "--prepare", "--dry-run", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["visual"]["download_required"] and result["asr"]["download_required"]


def test_offline_verification_fails_for_missing_model(monkeypatch, capsys):
    monkeypatch.setattr(projectctl, "model_inventory", lambda model=None: inventory(True, False))
    assert projectctl.main(["models", "--verify-offline", "--json"]) == 1
    output = capsys.readouterr()
    assert json.loads(output.out)["asr"]["offline_ready"] is False
    assert "offline verification failed" in output.err


def test_preprocess_checks_visual_before_ingest(monkeypatch, capsys):
    monkeypatch.setattr(projectctl, "model_inventory", lambda model=None: inventory(False, True))
    monkeypatch.setattr(projectctl, "ingest_report", projectctl.ingest_report)
    assert projectctl.main(["preprocess", "videos"]) == 1
    assert "models --prepare --visual" in capsys.readouterr().err


def test_index_parser_has_index_type():
    parsed = projectctl.parser().parse_args(["index", "--index-type", "hnsw"])
    assert parsed.index_type == "hnsw"


def test_ingest_config_preserves_documented_visual_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("VISUAL_SAMPLING_MODE", "sparse_shot")
    monkeypatch.setenv("VISUAL_GLOBAL_SAMPLE_SECONDS", "3.0")
    monkeypatch.setenv("VISUAL_DEDUP_ENABLED", "true")
    monkeypatch.setenv("VISUAL_DEDUP_THRESHOLD", "0.96")
    monkeypatch.setenv("VIDEO_EMBED_BATCH_SIZE", "7")
    monkeypatch.setenv("VIDEO_INDEX_TYPE", "hnsw")
    parsed = projectctl.parser().parse_args(
        ["ingest", "videos", "--processed-root", str(tmp_path)]
    )

    config = projectctl._video_ingest_config(parsed)

    assert config.visual_sampling_mode == "sparse_shot"
    assert config.visual_global_sample_seconds == 3.0
    assert config.visual_dedup_enabled is True
    assert config.visual_dedup_threshold == 0.96
    assert config.embed_batch_size == 7
    assert config.index_type == "hnsw"


def test_ingest_config_honors_visual_environment_in_subprocess(tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "VISUAL_SAMPLING_MODE": "sparse_shot",
            "VISUAL_GLOBAL_SAMPLE_SECONDS": "3.25",
            "VISUAL_DEDUP_ENABLED": "yes",
            "VISUAL_DEDUP_THRESHOLD": "0.96",
            "VIDEO_INDEX_TYPE": "hnsw",
        }
    )
    script = (
        "import json,projectctl; "
        "a=projectctl.parser().parse_args(['ingest','videos','--processed-root',"
        f"{str(tmp_path)!r}]); "
        "c=projectctl._video_ingest_config(a); "
        "print(json.dumps([c.visual_sampling_mode,c.visual_global_sample_seconds,"
        "c.visual_dedup_enabled,c.visual_dedup_threshold,c.index_type]))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert json.loads(completed.stdout) == ["sparse_shot", 3.25, True, 0.96, "hnsw"]


def test_invalid_environment_index_type_is_rejected(monkeypatch):
    monkeypatch.setenv("VIDEO_INDEX_TYPE", "bogus")
    with pytest.raises(SystemExit):
        projectctl.parser().parse_args(["index"])


def test_tesseract_language_spec_uses_only_installed_languages(monkeypatch):
    monkeypatch.setattr(projectctl, "tesseract_languages", lambda: {"eng"})
    assert projectctl.tesseract_language_spec() == "eng"

    monkeypatch.setattr(projectctl, "tesseract_languages", lambda: {"eng", "vie"})
    assert projectctl.tesseract_language_spec() == "eng+vie"

    monkeypatch.setattr(projectctl, "tesseract_languages", lambda: {"vie"})
    assert projectctl.tesseract_language_spec() == "vie"

    monkeypatch.setattr(projectctl, "tesseract_languages", lambda: set())
    with pytest.raises(RuntimeError, match="install tesseract eng or vie"):
        projectctl.tesseract_language_spec()


def test_parse_trake_json_and_pipe_events(tmp_path):
    path = tmp_path / "events.json"
    path.write_text('{"events":["one","two"]}')
    assert projectctl.parse_events(str(path)) == ["one", "two"]
    assert projectctl.parse_events("one | two") == ["one", "two"]


def test_asr_resume_does_not_construct_model(monkeypatch, tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"already processed")
    processed = tmp_path / "processed"
    evidence = processed / "video" / "asr.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("[]")
    import backend.app.video.text_backends as backends

    def unexpected(*args, **kwargs):
        raise AssertionError("Faster Whisper must not be constructed for a resumed ASR artifact")

    monkeypatch.setattr(backends, "FasterWhisperASRBackend", unexpected)
    monkeypatch.setattr(
        projectctl,
        "tesseract_languages",
        lambda: (_ for _ in ()).throw(
            AssertionError("ASR-only resume must not inspect OCR languages")
        ),
    )
    report = projectctl._text_pipeline(args(path=str(source), processed_root=str(processed),
        whisper_model="small", device="cpu", ocr_device=None, asr_device=None,
        asr_compute_type=None, limit=None), use_ocr=False, use_asr=True)

    assert report["failed"] == 0
    assert report["results"] == [{"video_id": "video", "status": "resumed",
        "ocr_count": 0, "asr_count": 0}]
