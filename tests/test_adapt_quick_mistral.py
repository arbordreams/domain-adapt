import os
import subprocess
import sys
import pytest


@pytest.mark.skipif(os.environ.get("RUN_SMOKE") != "1", reason="set RUN_SMOKE=1 to run adapt smoke")
def test_adapt_quick_mistral_runs() -> None:
    cmd = [
        sys.executable, "-m", "medical_tokalign.src.cli", "adapt",
        "--model_id", "mistralai/Mistral-7B-v0.3",
        "--top_k", "64", "--pivot", "64", "--warmup_steps", "0",
    ]
    proc = subprocess.run(cmd, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.returncode == 0, f"adapt failed:\n{proc.stdout}"


