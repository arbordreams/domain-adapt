from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple, Union
from collections import deque


RunnerCmd = Union[List[str], str]


def _pkg_root() -> str:
    # tools/ → pkg root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _repo_root() -> str:
    return os.path.abspath(os.path.join(_pkg_root(), ".."))


def _now() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


class Telemetry:
    def __init__(self, logs_dir: str):
        os.makedirs(logs_dir, exist_ok=True)
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.ts = ts
        self.text_log = os.path.join(logs_dir, f"pipeline_{ts}.log")
        self.jsonl = os.path.join(logs_dir, f"pipeline_{ts}.jsonl")
        self.summary = os.path.join(logs_dir, f"pipeline_{ts}_summary.json")
        self._text_fp = open(self.text_log, "a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write_text(self, line: str) -> None:
        with self._lock:
            self._text_fp.write(line)
            if not line.endswith("\n"):
                self._text_fp.write("\n")

    def write_event(self, event: Dict) -> None:
        with self._lock:
            with open(self.jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def write_summary(self, summary: Dict) -> None:
        with self._lock:
            with open(self.summary, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

    def close(self) -> None:
        try:
            self._text_fp.close()
        except Exception:
            pass


def _which(p: str) -> Optional[str]:
    return shutil.which(p)


def ensure_env_defaults() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HOME", os.path.join(_repo_root(), ".cache", "huggingface"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_repo_root(), ".cache", "huggingface", "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_repo_root(), ".cache", "huggingface", "transformers"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def preflight(telem: Telemetry, skip_bootstrap: bool) -> None:
    telem.write_text(f"[{_now()}] Preflight checks starting")
    # torch & cuda
    torch_ok = False
    try:
        import torch  # type: ignore
        torch_ok = bool(torch.cuda.is_available())
        telem.write_text(f"torch {getattr(torch, '__version__', 'unknown')} cuda={torch_ok}")
    except Exception as e:
        telem.write_text(f"[warn] torch import failed: {e}")

    # flash-attn
    flash_ok = False
    try:
        __import__("flash_attn")
        flash_ok = True
        telem.write_text("flash-attn import OK")
    except Exception as e:
        telem.write_text(f"[warn] flash-attn import failed: {e}")

    # hf_transfer
    try:
        __import__("hf_transfer")
        telem.write_text("hf_transfer import OK")
    except Exception as e:
        telem.write_text(f"[warn] hf_transfer import failed: {e}")

    # Prefer pinned dependency bootstrap for a consistent first run
    if not skip_bootstrap:
        telem.write_text("[info] Bootstrapping Python environment with pinned requirements")
        env = dict(os.environ)
        env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
        gpu_flag: list[str] = []
        try:
            if _which("nvidia-smi"):
                gpu_flag = ["--gpu-cu12"]
        except Exception:
            gpu_flag = []
        run_step(
            name="bootstrap",
            cmd=["/bin/bash", os.path.join(_pkg_root(), "scripts", "bootstrap_env.sh"), *gpu_flag],
            timeout_s=60 * 60,
            retries=0,
            telem=telem,
            extra_env=env,
        )
def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _acquire_lock(telem: Telemetry) -> Optional[str]:
    locks_dir = os.path.join(_pkg_root(), "runs", "locks")
    os.makedirs(locks_dir, exist_ok=True)
    lock_path = os.path.join(locks_dir, "pipeline.lock")
    try:
        if os.path.isfile(lock_path):
            with open(lock_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pid = int(data.get("pid", 0))
            if pid and _pid_is_running(pid):
                telem.write_text(f"[{_now()}] Another pipeline appears to be running (pid={pid}); aborting")
                return None
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "ts": _now()}, f)
        return lock_path
    except Exception:
        return None


def _release_lock(lock_path: Optional[str]) -> None:
    if lock_path and os.path.isfile(lock_path):
        try:
            os.remove(lock_path)
        except Exception:
            pass



def run_step(
    name: str,
    cmd: RunnerCmd,
    timeout_s: int,
    retries: int,
    telem: Telemetry,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[int, float]:
    started_at = time.time()
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    attempt = 0
    last_code = 0
    # Track recent output lines to classify errors as retryable vs fatal
    lines_tail: deque[str] = deque(maxlen=200)
    while True:
        attempt += 1
        telem.write_event({
            "ts": _now(), "step": name, "attempt": attempt, "status": "starting", "cmd": cmd,
        })
        telem.write_text(f"[{_now()}] >>> {name} (attempt {attempt})")

        # spawn process
        proc = subprocess.Popen(
            cmd if isinstance(cmd, list) else cmd,
            cwd=_repo_root(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=isinstance(cmd, str),
        )

        q: queue.Queue[str] = queue.Queue()
        stopped = threading.Event()

        def _pump():
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    q.put(line.rstrip("\n"))
            except Exception:
                pass
            finally:
                stopped.set()

        t = threading.Thread(target=_pump, daemon=True)
        t.start()

        code = None
        try:
            end_by = time.time() + timeout_s if timeout_s and timeout_s > 0 else None
            while True:
                # Drain any currently available lines quickly to reduce race with completion
                drained_any = False
                while True:
                    try:
                        line = q.get_nowait()
                        telem.write_text(line)
                        lines_tail.append(line)
                        drained_any = True
                    except queue.Empty:
                        break
                if not drained_any:
                    try:
                        # If nothing was immediately available, block briefly for next line
                        line = q.get(timeout=0.25)
                        telem.write_text(line)
                        lines_tail.append(line)
                    except queue.Empty:
                        pass
                if end_by and time.time() > end_by:
                    raise TimeoutError(f"step '{name}' timed out after {timeout_s}s")
                # Improved completion detection: wait for process exit and pump thread to finish,
                # then drain any remaining lines to avoid missing tail output.
                if proc.poll() is not None:
                    # Ensure the pump has observed EOF and exited
                    t.join(timeout=0.5)
                    # Final drain
                    while True:
                        try:
                            line = q.get_nowait()
                            telem.write_text(line)
                            lines_tail.append(line)
                        except queue.Empty:
                            break
                    code = int(proc.returncode)
                    break
            duration = time.time() - started_at
        except TimeoutError as e:
            duration = time.time() - started_at
            telem.write_text(f"[{_now()}] [timeout] {name}: {e}")
            # terminate process tree
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
            code = 124
        except Exception as e:
            duration = time.time() - started_at
            telem.write_text(f"[{_now()}] [error] {name}: {e}")
            code = 1

        last_code = int(code)
        telem.write_event({
            "ts": _now(), "step": name, "attempt": attempt,
            "status": "finished", "code": last_code, "duration_s": round(duration, 3),
        })

        # Classify errors to avoid unnecessary retries on fatal conditions
        # Retryable: common transient network/HF issues; Fatal: missing deps, not found, OOM, etc.
        lower_tail = "\n".join(lines_tail).lower()
        transient_markers = [
            "read timed out", "timed out", "connection reset", "connection reset by peer",
            "temporary failure in name resolution", "network is unreachable",
            "502 bad gateway", "503 service unavailable", "504 gateway timeout",
            "remotedisconnected", "ratelimit", "rate limit", "retrying", "connectionerror",
        ]
        fatal_markers = [
            "modulenotfounderror", "no module named", "importerror", "command not found",
            "filenotfounderror", "no such file or directory", "cuda initialization error",
            "cuda driver version", "out of memory", "permission denied", "killed",
        ]
        is_timeout = (last_code == 124)
        is_transient = is_timeout or any(m in lower_tail for m in transient_markers)
        is_fatal = any(m in lower_tail for m in fatal_markers)

        if last_code == 0:
            return last_code, duration
        if attempt > retries:
            return last_code, duration
        if is_fatal:
            telem.write_text(f"[{_now()}] [fatal] '{name}' appears non-retryable; not retrying")
            return last_code, duration
        # backoff
        time.sleep(min(60, 5 * attempt))


def _read_yaml(path: str) -> Dict:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _sum_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except Exception:
                pass
    return total


def _monitor_corpus(corpus_cfg: str, telem: Telemetry, stop_evt: threading.Event) -> None:
    cfg = _read_yaml(corpus_cfg)
    out_dir = cfg.get("output_dir") or os.path.join(_pkg_root(), "data", "biomed_corpus")
    target = int(cfg.get("target_total_bytes", 0))
    last_line = ""
    while not stop_evt.is_set():
        try:
            b = _sum_bytes(out_dir) if os.path.isdir(out_dir) else 0
            pct = (b / target * 100.0) if target > 0 else 0.0
            line = f"[corpus] {b/1e9:.2f} GB" + (f" / {target/1e9:.2f} GB ({pct:.1f}%)" if target > 0 else "")
            if line != last_line:
                telem.write_text(f"[{_now()}] {line}")
                telem.write_event({"ts": _now(), "progress": "corpus", "bytes": b, "target": target})
                last_line = line
        except Exception:
            pass
        stop_evt.wait(15.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Unattended MedTokAlign Orchestrator (RunPod-ready)")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2-7B")
    ap.add_argument("--corpus_config", type=str, default=os.path.join(_pkg_root(), "configs", "corpus_biomed.yaml"))
    ap.add_argument("--eval_config", type=str, default=os.path.join(_pkg_root(), "configs", "eval_medical.yaml"))
    ap.add_argument("--top_k", type=int, default=8192)
    ap.add_argument("--pivot", type=int, default=300)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--skip_bootstrap", action="store_true")
    ap.add_argument("--max_retries", type=int, default=1)
    ap.add_argument("--step_timeout", type=int, default=8 * 60 * 60)
    ap.add_argument("--tee", action="store_true")
    ap.add_argument("--env_pass_through", nargs="*", default=[])
    args = ap.parse_args()

    ensure_env_defaults()
    logs_dir = os.path.join(_pkg_root(), "runs", "logs")
    telem = Telemetry(logs_dir=logs_dir)
    summary: Dict[str, Dict] = {"started_at": _now(), "steps": {}}
    lock_path: Optional[str] = None
    try:
        # preflight + optional bootstrap
        preflight(telem, skip_bootstrap=bool(args.skip_bootstrap))

        # single-run lock
        lock_path = _acquire_lock(telem)
        if lock_path is None:
            telem.write_text(f"[{_now()}] Could not acquire pipeline lock; exiting")
            telem.write_summary({"locked": True, "finished_at": _now(), "exit_code": 1})
            sys.exit(1)

        # Step 1: prepare-data
        code, dur = run_step(
            name="prepare-data",
            # Use CLI directly via Python to avoid relying on removed script
            cmd=[sys.executable, "-m", "medical_tokalign.src.cli", "prepare-data", "--all"],
            timeout_s=max(30 * 60, args.step_timeout // 8),
            retries=args.max_retries,
            telem=telem,
        )
        if code != 0:
            raise SystemExit(code)
        summary["steps"]["prepare-data"] = {"code": code, "duration_s": round(dur, 2)}

        # Step 2: build-corpus with progress monitor
        stop_evt = threading.Event()
        mon = threading.Thread(target=_monitor_corpus, args=(args.corpus_config, telem, stop_evt), daemon=True)
        mon.start()
        try:
            env = {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "HF_DATASETS_TRUST_REMOTE_CODE": "1",
            }
            code, dur = run_step(
                name="build-corpus",
                cmd=[
                    sys.executable, "-m", "medical_tokalign.src.cli", "build-corpus",
                    "--config", os.path.abspath(args.corpus_config),
                ],
                timeout_s=max(6 * 60 * 60, args.step_timeout),
                retries=args.max_retries,
                telem=telem,
                extra_env=env,
            )
        finally:
            stop_evt.set()
        if code != 0:
            raise SystemExit(code)
        # Validate corpus completeness before proceeding to adapt.
        # Fail if clearly underfilled to avoid wasting compute on adaptation.
        try:
            cfg = _read_yaml(args.corpus_config)
            out_dir = cfg.get("output_dir") or os.path.join(_pkg_root(), "data", "biomed_corpus")
            target = int(cfg.get("target_total_bytes", 0))
            summ = {}
            p = os.path.join(out_dir, "summary.json")
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    summ = json.load(f)
            total = int(summ.get("total_bytes", 0)) if summ else _sum_bytes(out_dir)
            # Require at least 10% of target or a 5MB floor; warn if under target but above floor.
            min_required = max(5_000_000, int(0.10 * target)) if target > 0 else 5_000_000
            if total < min_required:
                telem.write_text(
                    f"[{_now()}] [error] corpus too small: {total} bytes < minimum {min_required} bytes"
                )
                raise SystemExit(3)
            if target > 0 and total < target:
                telem.write_text(f"[{_now()}] [warn] corpus under target: {total} < {target} bytes")
        except SystemExit:
            raise
        except Exception:
            pass
        summary["steps"]["build-corpus"] = {"code": code, "duration_s": round(dur, 2)}

        # Step 3: adapt (TokAlign-style)
        code, dur = run_step(
            name="adapt",
            cmd=[
                sys.executable, "-m", "medical_tokalign.src.cli", "adapt",
                "--model_id", args.model_id,
                "--top_k", str(int(args.top_k)),
                "--pivot", str(int(args.pivot)),
                "--warmup_steps", str(int(args.warmup_steps)),
            ],
            timeout_s=max(2 * 60 * 60, args.step_timeout // 2),
            retries=args.max_retries,
            telem=telem,
        )
        if code != 0:
            raise SystemExit(code)
        summary["steps"]["adapt"] = {"code": code, "duration_s": round(dur, 2)}

        # Step 4: eval
        code, dur = run_step(
            name="eval",
            cmd=[
                sys.executable, "-m", "medical_tokalign.src.cli", "eval",
                "--config", os.path.abspath(args.eval_config),
            ],
            timeout_s=max(2 * 60 * 60, args.step_timeout // 2),
            retries=args.max_retries,
            telem=telem,
        )
        if code != 0:
            raise SystemExit(code)
        summary["steps"]["eval"] = {"code": code, "duration_s": round(dur, 2)}

        summary["finished_at"] = _now()
        telem.write_text(f"[{_now()}] Pipeline completed successfully")
        telem.write_summary(summary)
        sys.exit(0)
    except SystemExit as e:
        summary["finished_at"] = _now()
        summary["exit_code"] = int(e.code) if isinstance(e.code, int) else 1
        telem.write_text(f"[{_now()}] Pipeline failed with exit code {summary['exit_code']}")
        telem.write_summary(summary)
        sys.exit(summary["exit_code"])  # rethrow
    except Exception as e:
        summary["finished_at"] = _now()
        summary["exit_code"] = 1
        telem.write_text(f"[{_now()}] Pipeline error: {e}")
        telem.write_summary(summary)
        sys.exit(1)
    finally:
        telem.close()
        _release_lock(lock_path)


if __name__ == "__main__":
    main()


