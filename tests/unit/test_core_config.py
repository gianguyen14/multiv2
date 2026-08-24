import json
import os
import subprocess
import sys


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
