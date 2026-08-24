import json
import os
import subprocess
import sys

import pytest


def test_siglip_configuration_honors_documented_environment(tmp_path):
    model_path = tmp_path / "local-siglip"
    environment = os.environ.copy()
    environment.update(
        {
            "SIGLIP_ENABLED": "false",
            "SIGLIP2_MODEL": str(model_path),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from backend.app.core.config import SIGLIP2_MODEL, SIGLIP_ENABLED; "
                "print(json.dumps([SIGLIP_ENABLED, SIGLIP2_MODEL]))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert json.loads(completed.stdout) == [False, str(model_path)]


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("SIGLIP_ENABLED", "sometimes", "must be one of"),
        ("SIGLIP2_MODEL", "   ", "must be a non-empty"),
    ],
)
def test_siglip_configuration_rejects_invalid_environment(name, value, error):
    environment = os.environ.copy()
    environment[name] = value
    completed = subprocess.run(
        [sys.executable, "-c", "import backend.app.core.config"],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert error in completed.stderr
