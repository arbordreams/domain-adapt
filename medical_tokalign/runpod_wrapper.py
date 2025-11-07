import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Optional


def configure_logging(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def project_root() -> str:
    # This file lives under medical_tokalign/, so repo root is parent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def verify_cuda(require_cuda: bool) -> None:
    if not require_cuda:
        return
    try:
        import torch  # type: ignore
    except Exception as e:
        logging.error("PyTorch not available: %s", e)
        sys.exit(1)
    if not torch.cuda.is_available():  # type: ignore
        logging.error("CUDA not available. Set --no-require-cuda to bypass.")
        sys.exit(1)


def verify_flash_attn(require_flash_attn: bool) -> None:
    if not require_flash_attn:
        return
    try:
        __import__("flash_attn")
    except Exception as e:
        logging.error("flash-attn import failed: %s", e)
        sys.exit(1)


def verify_hf_token(require_hf_token: bool) -> None:
    if not require_hf_token:
        return
    if ("HUGGINGFACE_TOKEN" not in os.environ) and ("HF_TOKEN" not in os.environ):
        logging.error("HF token required but not found in env (HF_TOKEN or HUGGINGFACE_TOKEN).")
        sys.exit(1)


def verify_glove_toolchain(check: bool) -> None:
    if not check:
        return
    if shutil.which("git") is None:
        logging.error("git not found in PATH; required to clone Stanford GloVe repository.")
        sys.exit(1)
    if shutil.which("make") is None:
        logging.error("make not found in PATH; required to build Stanford GloVe binaries.")
        sys.exit(1)


def run_cli_pipeline(model_id: str, corpus_config: str, eval_config: str, top_k: int, pivot: int, warmup_steps: int, timeout_s: int) -> int:
    cmd = [
        sys.executable,
        "-m",
        "medical_tokalign.src.cli",
        "pipeline",
        "--model_id",
        model_id,
        "--corpus_config",
        corpus_config,
        "--eval_config",
        eval_config,
        "--top_k",
        str(int(top_k)),
        "--pivot",
        str(int(pivot)),
        "--warmup_steps",
        str(int(warmup_steps)),
    ]
    logging.info("Executing CLI pipeline: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, timeout=int(timeout_s), check=False)
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        logging.error("Pipeline timed out after %s seconds", timeout_s)
        return 124


def find_latest_adapted_run(runs_root: str) -> Optional[str]:
    if not os.path.isdir(runs_root):
        return None
    candidates = sorted(glob.glob(os.path.join(runs_root, "*")))
    candidates = [d for d in candidates if os.path.isdir(d)]
    return candidates[-1] if candidates else None


def verify_artifacts(run_dir: str) -> bool:
    if not run_dir or not os.path.isdir(run_dir):
        return False
    model_dir = os.path.join(run_dir, "model")
    tok_dir = os.path.join(run_dir, "tokenizer")
    align_json = os.path.join(run_dir, "align_matrix.json")
    ok = True
    if not os.path.isdir(model_dir):
        logging.error("Missing model dir: %s", model_dir)
        ok = False
    if not os.path.isdir(tok_dir):
        logging.error("Missing tokenizer dir: %s", tok_dir)
        ok = False
    if not os.path.isfile(align_json):
        logging.error("Missing alignment file: %s", align_json)
        ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description="RunPod wrapper: execute MedTokAlign CLI pipeline with validations.")
    ap.add_argument("--timeout", type=int, default=4 * 60 * 60, help="Pipeline timeout in seconds")
    ap.add_argument("--no-require-cuda", action="store_true", help="Do not require CUDA availability")
    ap.add_argument("--require-flash-attn", action="store_true", help="Require flash-attn import to succeed")
    ap.add_argument("--require-hf-token", action="store_true", help="Require HF token in env (HF_TOKEN or HUGGINGFACE_TOKEN)")
    ap.add_argument("--verify-glove-tools", action="store_true", help="Fail early if git/make are not installed")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2-7B")
    ap.add_argument("--top_k", type=int, default=8192)
    ap.add_argument("--pivot", type=int, default=300)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--corpus_config", type=str, default=os.path.join("medical_tokalign", "configs", "corpus_biomed.yaml"))
    ap.add_argument("--eval_config", type=str, default=os.path.join("medical_tokalign", "configs", "eval_medical.yaml"))
    ap.add_argument(
        "--log-file",
        type=str,
        default=os.path.join("medical_tokalign", "runs", "logs", f"runpod_{time.strftime('%Y%m%d_%H%M%S')}.log"),
        help="Path to log file",
    )
    args = ap.parse_args()

    # Best-effort environment guards
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    if not os.environ.get("HF_TOKEN"):
        if os.environ.get("HUGGINGFACE_TOKEN"):
            os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", os.environ["HF_TOKEN"])  # convenience

    configure_logging(args.log_file)
    root = project_root()
    logging.info("Project root: %s", root)

    verify_cuda(not args.no_require_cuda)
    verify_flash_attn(args.require_flash_attn)
    verify_hf_token(args.require_hf_token)
    verify_glove_toolchain(args.verify_glove_tools)

    # Execute CLI pipeline
    rc = run_cli_pipeline(
        model_id=args.model_id,
        corpus_config=os.path.join(root, args.corpus_config) if not os.path.isabs(args.corpus_config) else args.corpus_config,
        eval_config=os.path.join(root, args.eval_config) if not os.path.isabs(args.eval_config) else args.eval_config,
        top_k=int(args.top_k),
        pivot=int(args.pivot),
        warmup_steps=int(args.warmup_steps),
        timeout_s=int(args.timeout),
    )
    if rc != 0:
        logging.error("CLI pipeline failed with return code %s", rc)
        sys.exit(rc)

    # Verify artifacts from latest run
    runs_root = os.path.join(root, "medical_tokalign", "runs", "tokenizer_adapt")
    latest = find_latest_adapted_run(runs_root)
    if not latest:
        logging.error("No tokenizer_adapt runs found in %s", runs_root)
        sys.exit(3)
    if not verify_artifacts(latest):
        logging.error("Artifacts verification failed for %s", latest)
        sys.exit(3)

    logging.info("Success. Latest adapted run: %s", latest)
    print(latest)
    sys.exit(0)


if __name__ == "__main__":
    main()


