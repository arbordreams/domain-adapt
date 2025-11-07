from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List, Tuple


def bootstrap_diff(
    base: List[float],
    adapted: List[float],
    n_samples: int = 1000,
    seed: int = 17,
) -> Dict[str, float]:
    """Bootstrap CI for mean(adapted-base).

    Inputs are per-sample correctness (0/1) or any scalar scores.
    """
    if not base or not adapted or len(base) != len(adapted):
        return {"delta_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value_two_sided": 1.0}

    rng = random.Random(int(seed))
    n = len(base)
    deltas: List[float] = []
    base_mean = sum(base) / n
    adapt_mean = sum(adapted) / n
    delta_obs = adapt_mean - base_mean
    for _ in range(int(n_samples)):
        idxs = [rng.randrange(0, n) for _ in range(n)]
        b = [base[i] for i in idxs]
        a = [adapted[i] for i in idxs]
        deltas.append((sum(a) / n) - (sum(b) / n))
    s = sorted(deltas)
    lo = s[int(0.025 * (len(s) - 1))]
    hi = s[int(0.975 * (len(s) - 1))]
    # two-sided p-value under bootstrap distribution
    le = sum(1 for d in deltas if d <= 0) / len(deltas)
    ge = sum(1 for d in deltas if d >= 0) / len(deltas)
    p_two = 2.0 * min(le, ge)
    return {"delta_mean": float(delta_obs), "ci_low": float(lo), "ci_high": float(hi), "p_value_two_sided": float(min(1.0, max(0.0, p_two)))}


def save_stats(run_dir: str, stats: Dict[str, Dict[str, float]]) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    # simple human-readable diagnostics
    lines: List[str] = []
    for ds, st in stats.items():
        lines.append(f"{ds}: delta_mean={st.get('delta_mean', float('nan')):.4f}, CI95=[{st.get('ci_low', float('nan')):.4f}, {st.get('ci_high', float('nan')):.4f}], p(two)={st.get('p_value_two_sided', 1.0):.4f}")
    with open(os.path.join(run_dir, "diagnostics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


