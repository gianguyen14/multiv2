from pathlib import Path


def test_dockerfile_packages_streaming_benchmark_script():
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "Dockerfile").read_text()
    script = root / "scripts" / "benchmark_qwen_streaming.py"
    assert script.is_file()
    assert "COPY scripts scripts" in dockerfile
    assert "scripts/benchmark_qwen_streaming.py" in script.read_text()
