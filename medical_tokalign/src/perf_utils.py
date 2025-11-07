import contextlib
import os
import torch


def configure_torch(allow_tf32: bool = True, matmul_precision: str = "high") -> None:
    """Set global torch performance flags for H100 runs."""
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision(str(matmul_precision))
    except Exception:
        pass


def maybe_compile(model: torch.nn.Module | None, enabled: bool = True, mode: str = "max-autotune"):
    """Compile the model with torch.compile when enabled (PyTorch 2.4+)."""
    if not enabled or model is None:
        return model
    try:
        compiled = torch.compile(model, mode=mode, fullgraph=False, dynamic=False)
        return compiled
    except Exception:
        return model


@contextlib.contextmanager
def cuda_graph_guard(enabled: bool = False):
    """A simple guard to hint CUDA Graphs usage; no-op if disabled.
    In practice, integrate with capture/replay in custom inference loops if needed.
    """
    if not enabled:
        yield
        return
    g = torch.cuda.CUDAGraph()
    try:
        yield g
    finally:
        del g


def default_env_setup() -> None:
    """Set recommended environment variables for memory behavior."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def estimate_vllm_max_batch_tokens(h100_vram_gb: int = 80) -> int:
    """Return a conservative max_num_batched_tokens for vLLM by H100 VRAM size."""
    if h100_vram_gb >= 96:
        return 131072
    if h100_vram_gb >= 80:
        return 65536
    return 32768


