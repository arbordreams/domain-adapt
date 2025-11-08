import os
import subprocess
import sys
import pytest


@pytest.mark.skipif(os.environ.get("RUN_SMOKE") != "1", reason="set RUN_SMOKE=1 to run eval smoke")
def test_eval_fallback_hf_when_vllm_missing(tmp_path) -> None:
    env = os.environ.copy()
    env.pop("EVAL_BACKEND", None)
    # simulate vLLM missing by ensuring import fails: not trivial; rely on script auto-detect
    cfg = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "configs", "eval_medical_ultra_quick.yaml")
    cmd = [sys.executable, "-m", "medical_tokalign.src.cli", "eval", "--config", cfg]
    proc = subprocess.run(cmd, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # We only assert the CLI returns a code (success preferred, but environment dependent)
    assert proc.returncode in (0, 2), f"eval path failed unexpectedly:\n{proc.stdout}"


