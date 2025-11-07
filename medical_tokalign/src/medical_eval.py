from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple
import math

import torch
import torch.nn.functional as F
import yaml

from .datasets_medical import (
    load_pubmedqa,
    load_medqa,
    load_mednli,
    load_ncbi_disease,
    load_bc5cdr,
    load_perplexity_corpus,
    format_pubmedqa_prompt,
    format_medqa_prompt,
    format_mednli_prompt,
    extract_ner_gold_spans,
    format_ner_prompt,
    load_medmcqa,
    format_medmcqa_prompt,
    load_mmlu_medical,
    format_mmlu_prompt,
)
from .llama_utils import (
    find_latest_adapted_artifacts,
    hf_generate,
    load_hf_model_and_tokenizer,
    load_vllm_engine,
    vllm_generate,
)
from .metrics_medical import accuracy, extract_choice, ner_span_f1, perplexity


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(rows: List[Tuple[str, str, str, float]], path: str) -> None:
    # rows of (variant, dataset, metric, value)
    with open(path, "w", encoding="utf-8") as f:
        f.write("variant,dataset,metric,value\n")
        for v, d, m, val in rows:
            f.write(f"{v},{d},{m},{val}\n")


def _build_letter_token_candidates(tok) -> Dict[str, List[int]]:
    # Collect single-token candidates for A–D with/without leading space and lowercase
    cand: Dict[str, List[int]] = {c: [] for c in "ABCD"}
    for c in "ABCD":
        for s in [c, c.lower(), f" {c}", f" {c.lower()}"]:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                if ids[0] not in cand[c]:
                    cand[c].append(ids[0])
    return cand


def _score_suffix_loglik_batch(model, tok, prompts: List[str], suffix: str, max_length: int | None, batch_size: int) -> List[float]:
    device = next(model.parameters()).device
    results: List[float] = []
    idx = 0
    with torch.no_grad():
        while idx < len(prompts):
            cur_bs = min(batch_size, len(prompts) - idx)
            while cur_bs > 0:
                try:
                    sub = prompts[idx: idx + cur_bs]
                    # Build combined sequences per sample to ensure suffix fully present
                    input_ids_list = []
                    suffix_lens = []
                    for p in sub:
                        p_ids = tok(p, add_special_tokens=False).input_ids
                        s_ids = tok(suffix, add_special_tokens=False).input_ids
                        if max_length:
                            max_len = int(max_length)
                        else:
                            max_len = len(p_ids) + len(s_ids)
                        keep_prompt = max(0, max_len - len(s_ids))
                        p_tail = p_ids[-keep_prompt:] if keep_prompt < len(p_ids) else p_ids
                        combined = p_tail + s_ids
                        input_ids_list.append(torch.tensor(combined, dtype=torch.long))
                        suffix_lens.append(len(s_ids))

                    # Left-pad to max length
                    max_seq = max(t.size(0) for t in input_ids_list)
                    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                    batch_ids = torch.full((len(input_ids_list), max_seq), pad_id, dtype=torch.long)
                    for i, t in enumerate(input_ids_list):
                        batch_ids[i, -t.size(0):] = t
                    attn = (batch_ids != pad_id).long()
                    batch_ids = batch_ids.to(device)
                    attn = attn.to(device)

                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        out = model(input_ids=batch_ids, attention_mask=attn)
                    logits = out.logits  # [B, T, V]
                    logprobs = torch.log_softmax(logits, dim=-1)

                    # For each sample, sum log-probs of suffix tokens
                    for i in range(batch_ids.size(0)):
                        seq_len = int(attn[i].sum().item())
                        s_len = suffix_lens[i]
                        if s_len == 0 or seq_len <= 1:
                            results.append(float("-inf"))
                            continue
                        # positions in logits used to predict next token (shifted by 1)
                        start_pos = seq_len - s_len - 1
                        if start_pos < 0:
                            results.append(float("-inf"))
                            continue
                        ll = 0.0
                        for j in range(s_len):
                            pos = start_pos + j
                            next_token = int(batch_ids[i, -s_len + j].item())  # last s_len tokens are suffix
                            ll += float(logprobs[i, pos, next_token].item())
                        results.append(ll)
                    idx += cur_bs
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    cur_bs = max(1, cur_bs // 2)
            if cur_bs == 0:
                # If even bs=1 fails, append -inf and move on
                results.append(float("-inf"))
                idx += 1
    return results


def predict_mcq_letters_ll_hf(
    model,
    tok,
    prompts: List[str],
    batch_size: int = 16,
    max_length: int | None = None,
) -> List[str]:
    # Score letters A–D via conditional log-likelihood over simple variants
    letters = list("ABCD")
    variants = [" {L}", "{L}", "({L})", "[{L}]"]
    # For each letter, compute best score across variants
    per_letter_scores: Dict[str, List[float]] = {L: [float("-inf")] * len(prompts) for L in letters}
    for L in letters:
        best = [float("-inf")] * len(prompts)
        for v in variants:
            suf = v.format(L=L)
            scores = _score_suffix_loglik_batch(model, tok, prompts, suf, max_length, batch_size)
            # take max across variants
            for i, sc in enumerate(scores):
                if sc > best[i]:
                    best[i] = sc
        per_letter_scores[L] = best
    # Select argmax letter per prompt
    results: List[str] = []
    for i in range(len(prompts)):
        best_L = ""
        best_sc = float("-inf")
        for L in letters:
            sc = per_letter_scores[L][i]
            if sc > best_sc:
                best_sc = sc
                best_L = L
        results.append(best_L)
    return results


def main():
    ap = argparse.ArgumentParser(description="MedTokAlign - Medical Evaluation Orchestrator")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Optional seeding for reproducibility
    try:
        if bool(cfg.get("seed_everything", False)):
            import random as _rnd
            _seed = int(cfg.get("random_seed", 17))
            _rnd.seed(_seed)
            try:
                import numpy as _np
                _np.random.seed(_seed)
            except Exception:
                pass
            try:
                import torch as _torch
                _torch.manual_seed(_seed)
                if _torch.cuda.is_available():
                    _torch.cuda.manual_seed_all(_seed)
            except Exception:
                pass
    except Exception:
        pass

    out_root = cfg.get("output_dir", "medical_tokalign/runs/medical_eval")
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, ts)
    ensure_dir(run_dir)

    # Discover adapted artifacts (no silent override)
    search_dir = cfg.get("alignment", {}).get("search_dir", "medical_tokalign/runs/tokenizer_adapt")
    adapted = find_latest_adapted_artifacts(search_dir) if os.path.isdir(search_dir) else None

    model_id = cfg.get("model_id")
    # Backend precedence: env override > config (default vllm)
    backend_env = os.environ.get("EVAL_BACKEND")
    backend = (backend_env or cfg.get("eval_backend", "vllm")).lower()
    precision = cfg.get("precision", "bf16")
    attn_impl = cfg.get("attn_impl", "flash_attention_2")
    compile_flag = bool(cfg.get("compile", True))
    grad_ckpt = bool(cfg.get("grad_ckpt", False))
    allow_tf32 = bool(cfg.get("allow_tf32", True))

    align_cfg = cfg.get("alignment", {})
    compare_mode = str(align_cfg.get("compare_mode", "base_vs_adapted")).lower()
    use_adapted_model = bool(align_cfg.get("use_adapted_model", False))
    use_adapted_tokenizer = bool(align_cfg.get("use_adapted_tokenizer", False))

    variants: List[Dict[str, str | None]] = []
    if compare_mode == "base_vs_adapted":
        variants.append({"name": "base", "model": model_id, "tokenizer": None})
        if adapted and adapted.model_dir:
            variants.append({"name": "adapted", "model": adapted.model_dir, "tokenizer": adapted.tokenizer_dir})
        elif adapted:
            print("[MedTokAlign] Adapted artifacts directory found but appears incomplete; proceeding with base only.")
    else:
        if use_adapted_model and adapted and adapted.model_dir:
            variants.append({"name": "adapted", "model": adapted.model_dir, "tokenizer": adapted.tokenizer_dir if use_adapted_tokenizer else None})
        else:
            variants.append({"name": "base", "model": model_id, "tokenizer": None})
            if adapted and adapted.model_dir:
                print("[MedTokAlign] Adapted artifacts detected but disabled by config (use_adapted_model=false).")

    # Paths
    # Resolve data root from config or default to package-relative
    cfg_data_root = cfg.get("data_root")
    if cfg_data_root and isinstance(cfg_data_root, str):
        data_root = os.path.abspath(cfg_data_root)
    else:
        data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "medical")
    bench_dir = os.path.join(data_root, "benchmarks")
    proc_dir = os.path.join(data_root, "processed")

    # Dataset config helpers
    def _ds_cfg(name: str) -> dict:
        ds = cfg.get("datasets", {}).get(name)
        if isinstance(ds, bool):
            return {"enabled": ds}
        return ds or {"enabled": False}

    def _ds_enabled(name: str) -> bool:
        return bool(_ds_cfg(name).get("enabled", False))

    def _ds_split(name: str) -> str:
        return str(_ds_cfg(name).get("split", "test"))

    def _ds_limit(name: str) -> int:
        try:
            return int(_ds_cfg(name).get("limit", 0))
        except Exception:
            return 0

    # Run per-variant evaluation and return early; supersedes legacy single-variant path
    summary: Dict[str, Dict[str, float]] = {}
    variant_correct: Dict[str, Dict[str, List[int]]] = {}
    for var in variants:
        var_dir = os.path.join(run_dir, var["name"]) if len(variants) > 1 else run_dir
        ensure_dir(var_dir)

        # Backend init per variant
        eff_backend = backend
        engine = None
        hf_model = None
        hf_tok = None
        if eff_backend == "vllm":
            vllm_cfg = cfg.get("vllm", {})
            try:
                print(f"[MedTokAlign][vLLM] cfg={json.dumps(vllm_cfg)}")
                engine = load_vllm_engine(
                    model=var["model"],
                    tokenizer=var.get("tokenizer") or var["model"],
                    tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
                    gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.92)),
                    max_model_len=int(vllm_cfg.get("max_model_len", 8192)),
                    enforce_eager=bool(vllm_cfg.get("enforce_eager", False)),
                    trust_remote_code=bool(vllm_cfg.get("trust_remote_code", False)),
                    max_num_batched_tokens=int(vllm_cfg.get("max_batch_tokens", 0)) if vllm_cfg.get("max_batch_tokens") else None,
                )
            except Exception as e:
                print(f"[MedTokAlign] vLLM unavailable or failed to initialize ({e}). Falling back to HF backend.")
                eff_backend = "hf"
        if eff_backend != "vllm":
            hf_model, hf_tok = load_hf_model_and_tokenizer(
                model_id_or_path=var["model"],
                precision=precision,
                attn_impl=attn_impl,
                compile_model=compile_flag,
                grad_ckpt=grad_ckpt,
                allow_tf32=allow_tf32,
                max_model_len=int(cfg.get("hf", {}).get("max_model_len", 8192)),
                tokenizer_dir_override=var.get("tokenizer"),
                device_map=cfg.get("hf", {}).get("device_map", "auto"),
            )

        # Evaluation accumulators
        metrics: Dict[str, float] = {}
        samples: List[str] = []
        per_dataset_outputs: Dict[str, List[Dict[str, str]]] = {}
        per_dataset_correct: Dict[str, List[int]] = {}

        # Lightweight throughput logging (prompts/s and approx tokens/s)
        tok_for_stats = None  # lazy
        def _ensure_tok_for_stats():
            nonlocal tok_for_stats, hf_tok
            if hf_tok is not None:
                tok_for_stats = hf_tok
                return
            if tok_for_stats is None:
                try:
                    from transformers import AutoTokenizer as _ATok
                    tok_for_stats = _ATok.from_pretrained(var["model"], use_fast=True)
                except Exception:
                    tok_for_stats = None

        def do_generate(prompts: List[str]) -> List[str]:
            gen = cfg.get("gen", {})
            t0 = time.time()
            if eff_backend == "vllm":
                outs = vllm_generate(
                    engine,
                    prompts,
                    max_new_tokens=int(gen.get("max_new_tokens", 64)),
                    temperature=float(gen.get("temperature", 0.0)),
                    top_p=float(gen.get("top_p", 1.0)),
                    top_k=int(gen.get("top_k", 0)),
                    stop=gen.get("stop", []) or [],
                )
                dt = max(1e-6, time.time() - t0)
                _ensure_tok_for_stats()
                tok_cnt = 0
                if tok_for_stats is not None:
                    try:
                        for t in outs:
                            tok_cnt += len(tok_for_stats(t, add_special_tokens=False).input_ids)
                    except Exception:
                        pass
                print(f"[MedTokAlign][vLLM] gen: prompts={len(prompts)} time={dt:.2f}s prompts/s={len(prompts)/dt:.2f} tokens/s={(tok_cnt/dt) if tok_cnt else -1:.2f}")
                return outs
            else:
                outs = hf_generate(
                    hf_model,
                    hf_tok,
                    prompts,
                    max_new_tokens=int(gen.get("max_new_tokens", 64)),
                    temperature=float(gen.get("temperature", 0.0)),
                    top_p=float(gen.get("top_p", 1.0)),
                    top_k=int(gen.get("top_k", 0)),
                    stop=gen.get("stop", []) or [],
                    batch_size=int(cfg.get("hf", {}).get("per_device_batch_size", 4)),
                )
                dt = max(1e-6, time.time() - t0)
                _ensure_tok_for_stats()
                tok_cnt = 0
                if tok_for_stats is not None:
                    try:
                        for t in outs:
                            tok_cnt += len(tok_for_stats(t, add_special_tokens=False).input_ids)
                    except Exception:
                        pass
                print(f"[MedTokAlign][HF] gen: prompts={len(prompts)} time={dt:.2f}s prompts/s={len(prompts)/dt:.2f} tokens/s={(tok_cnt/dt) if tok_cnt else -1:.2f}")
                return outs

        # PubMedQA
        if _ds_enabled("pubmedqa"):
            ds = load_pubmedqa(bench_dir)
            split = _ds_split("pubmedqa")
            test = (ds.get(split) or [])
            lim = _ds_limit("pubmedqa")
            if lim > 0:
                test = test[:lim]
            if test:
                prompts, labels = [], []
                for ex in test:
                    p, y = format_pubmedqa_prompt(ex)
                    prompts.append(p); labels.append(y)
                preds = do_generate(prompts)
                preds_mapped = []
                for t in preds:
                    low = t.strip().lower()
                    mapped = "maybe"
                    if low.startswith("yes"):
                        mapped = "yes"
                    elif low.startswith("no"):
                        mapped = "no"
                    elif low.startswith("maybe"):
                        mapped = "maybe"
                    preds_mapped.append(mapped)
                acc = accuracy(preds_mapped, labels)
                metrics["pubmedqa_acc"] = acc
                per_dataset_outputs["pubmedqa"] = [{"prompt": pr, "raw": prd, "parsed": pm, "label": y} for pr, prd, pm, y in zip(prompts, preds, preds_mapped, labels)]
                samples.append(f"PubMedQA: acc={acc:.3f}")
                try:
                    per_dataset_correct["pubmedqa"] = [1 if (pm == y) else 0 for pm, y in zip(preds_mapped, labels)]
                except Exception:
                    pass

        # MedQA (USMLE)
        if _ds_enabled("medqa_usmle"):
            ds = load_medqa(bench_dir)
            split = _ds_split("medqa_usmle")
            test = (ds.get(split) or [])
            lim = _ds_limit("medqa_usmle")
            if lim > 0:
                test = test[:lim]
            if test:
                prompts, labels = [], []
                for ex in test:
                    p, y = format_medqa_prompt(ex)
                    prompts.append(p); labels.append(y)
                # Prefer MCQ scoring via conditional LL
                if hf_model is None or hf_tok is None:
                    preds = do_generate(prompts)
                    letters = [extract_choice(t) for t in preds]
                else:
                    letters = predict_mcq_letters_ll_hf(
                        hf_model,
                        hf_tok,
                        prompts,
                        batch_size=int(cfg.get("hf", {}).get("per_device_batch_size", 16)),
                        max_length=int(cfg.get("hf", {}).get("max_model_len", 4096)),
                    )
                    # Fallback parse for any empties
                    if any(not x for x in letters):
                        fallbacks = do_generate([prompts[i] for i, l in enumerate(letters) if not l])
                        for (i, _), t in zip([(idx, l) for idx, l in enumerate(letters) if not l], fallbacks):
                            letters[i] = extract_choice(t)
                parsed_ok = sum(1 for x in letters if x)
                print(f"[MedTokAlign][MedQA] parsed_choices={parsed_ok}/{len(letters)}")
                acc = accuracy(letters, labels)
                metrics["medqa_acc"] = acc
                per_dataset_outputs["medqa"] = [{"prompt": pr, "pred": le, "label": y} for pr, le, y in zip(prompts, letters, labels)]
                samples.append(f"MedQA: acc={acc:.3f}")
                try:
                    per_dataset_correct["medqa"] = [1 if (le == y) else 0 for le, y in zip(letters, labels)]
                except Exception:
                    pass

        # MedMCQA
        if _ds_enabled("medmcqa"):
            ds = load_medmcqa(bench_dir)
            split = _ds_split("medmcqa") or "validation"
            test = (ds.get(split) or [])
            lim = _ds_limit("medmcqa")
            if lim > 0:
                test = test[:lim]
            if test:
                prompts, labels = [], []
                for ex in test:
                    p, y = format_medmcqa_prompt(ex)
                    prompts.append(p); labels.append(y)
                if hf_model is None or hf_tok is None:
                    preds = do_generate(prompts)
                    letters = [extract_choice(t) for t in preds]
                else:
                    letters = predict_mcq_letters_ll_hf(
                        hf_model,
                        hf_tok,
                        prompts,
                        batch_size=int(cfg.get("hf", {}).get("per_device_batch_size", 16)),
                        max_length=int(cfg.get("hf", {}).get("max_model_len", 4096)),
                    )
                    if any(not x for x in letters):
                        fallbacks = do_generate([prompts[i] for i, l in enumerate(letters) if not l])
                        for (i, _), t in zip([(idx, l) for idx, l in enumerate(letters) if not l], fallbacks):
                            letters[i] = extract_choice(t)
                parsed_ok = sum(1 for x in letters if x)
                print(f"[MedTokAlign][MedMCQA] parsed_choices={parsed_ok}/{len(letters)}")
                acc = accuracy(letters, labels)
                metrics["medmcqa_acc"] = acc
                per_dataset_outputs["medmcqa"] = [{"prompt": pr, "pred": le, "label": y} for pr, le, y in zip(prompts, letters, labels)]
                samples.append(f"MedMCQA: acc={acc:.3f}")
                try:
                    per_dataset_correct["medmcqa"] = [1 if (le == y) else 0 for le, y in zip(letters, labels)]
                except Exception:
                    pass

        # MMLU - medical subset
        if _ds_enabled("mmlu_medical"):
            ds = load_mmlu_medical(bench_dir, subjects=_ds_cfg("mmlu_medical").get("subjects", []))
            split = _ds_split("mmlu_medical")
            test = (ds.get(split) or [])
            lim = _ds_limit("mmlu_medical")
            if lim > 0:
                test = test[:lim]
            if test:
                prompts, labels = [], []
                for ex in test:
                    p, y = format_mmlu_prompt(ex)
                    prompts.append(p); labels.append(y)
                if hf_model is None or hf_tok is None:
                    preds = do_generate(prompts)
                    letters = [extract_choice(t) for t in preds]
                else:
                    letters = predict_mcq_letters_ll_hf(
                        hf_model,
                        hf_tok,
                        prompts,
                        batch_size=int(cfg.get("hf", {}).get("per_device_batch_size", 16)),
                        max_length=int(cfg.get("hf", {}).get("max_model_len", 4096)),
                    )
                    if any(not x for x in letters):
                        fallbacks = do_generate([prompts[i] for i, l in enumerate(letters) if not l])
                        for (i, _), t in zip([(idx, l) for idx, l in enumerate(letters) if not l], fallbacks):
                            letters[i] = extract_choice(t)
                parsed_ok = sum(1 for x in letters if x)
                print(f"[MedTokAlign][MMLU-medical] parsed_choices={parsed_ok}/{len(letters)}")
                acc = accuracy(letters, labels)
                metrics["mmlu_medical_acc"] = acc
                per_dataset_outputs["mmlu_medical"] = [{"prompt": pr, "pred": le, "label": y} for pr, le, y in zip(prompts, letters, labels)]
                samples.append(f"MMLU-medical: acc={acc:.3f}")
                try:
                    per_dataset_correct["mmlu_medical"] = [1 if (le == y) else 0 for le, y in zip(letters, labels)]
                except Exception:
                    pass

        # MedNLI
        if _ds_enabled("mednli"):
            ds = load_mednli(bench_dir)
            split = _ds_split("mednli")
            test = (ds.get(split) or [])
            lim = _ds_limit("mednli")
            if lim > 0:
                test = test[:lim]
            if test:
                prompts, labels = [], []
                for ex in test:
                    p, y = format_mednli_prompt(ex)
                    prompts.append(p); labels.append(y)
                preds = do_generate(prompts)
                preds_mapped = []
                for t in preds:
                    low = t.strip().lower()
                    if low.startswith("entailment"):
                        preds_mapped.append("entailment")
                    elif low.startswith("contradiction"):
                        preds_mapped.append("contradiction")
                    else:
                        preds_mapped.append("neutral")
                acc = accuracy(preds_mapped, labels)
                metrics["mednli_acc"] = acc
                per_dataset_outputs["mednli"] = [{"prompt": pr, "raw": prd, "parsed": pm, "label": y} for pr, prd, pm, y in zip(prompts, preds, preds_mapped, labels)]
                samples.append(f"MedNLI: acc={acc:.3f}")
                try:
                    per_dataset_correct["mednli"] = [1 if (pm == y) else 0 for pm, y in zip(preds_mapped, labels)]
                except Exception:
                    pass

        # NER: NCBI-Disease
        if _ds_enabled("ncbi_disease"):
            ds = load_ncbi_disease(bench_dir)
            split = _ds_split("ncbi_disease")
            test = (ds.get(split) or [])
            lim = _ds_limit("ncbi_disease")
            if lim > 0:
                test = test[:lim]
            if test:
                def _extract_text(ex: dict) -> str:
                    for k in ("text", "document", "passages", "context", "abstract", "sentence"):
                        v = ex.get(k)
                        if isinstance(v, list):
                            v = "\n".join(map(str, v))
                        if isinstance(v, str) and v:
                            return v
                    return ""
                prompts, gold_lists = [], []
                for ex in test:
                    txt = _extract_text(ex)
                    prompts.append(format_ner_prompt(txt))
                    gold_lists.append(extract_ner_gold_spans(ex))
                preds = do_generate(prompts)
                pred_lists = []
                for t in preds:
                    lines = [ln.strip() for ln in str(t).splitlines()]
                    pred_lists.append([ln for ln in lines if ln])
                f1s = ner_span_f1(pred_lists, gold_lists)
                metrics["ncbi_disease_f1"] = float(f1s.get("f1", 0.0))
                per_dataset_outputs["ncbi_disease"] = [{"prompt": pr, "pred_spans": ps, "gold_spans": gs} for pr, ps, gs in zip(prompts, pred_lists, gold_lists)]
                samples.append(f"NCBI-Disease: f1={metrics['ncbi_disease_f1']:.3f}")

        # NER: BC5CDR
        if _ds_enabled("bc5cdr"):
            ds = load_bc5cdr(bench_dir)
            split = _ds_split("bc5cdr")
            test = (ds.get(split) or [])
            lim = _ds_limit("bc5cdr")
            if lim > 0:
                test = test[:lim]
            if test:
                def _extract_text2(ex: dict) -> str:
                    for k in ("text", "document", "passages", "context", "abstract", "sentence"):
                        v = ex.get(k)
                        if isinstance(v, list):
                            v = "\n".join(map(str, v))
                        if isinstance(v, str) and v:
                            return v
                    return ""
                prompts, gold_lists = [], []
                for ex in test:
                    txt = _extract_text2(ex)
                    prompts.append(format_ner_prompt(txt))
                    gold_lists.append(extract_ner_gold_spans(ex))
                preds = do_generate(prompts)
                pred_lists = []
                for t in preds:
                    lines = [ln.strip() for ln in str(t).splitlines()]
                    pred_lists.append([ln for ln in lines if ln])
                f1s = ner_span_f1(pred_lists, gold_lists)
                metrics["bc5cdr_f1"] = float(f1s.get("f1", 0.0))
                per_dataset_outputs["bc5cdr"] = [{"prompt": pr, "pred_spans": ps, "gold_spans": gs} for pr, ps, gs in zip(prompts, pred_lists, gold_lists)]
                samples.append(f"BC5CDR: f1={metrics['bc5cdr_f1']:.3f}")

        # Perplexity (HF path)
        ppl_src = cfg.get("datasets", {}).get("perplexity_corpus")
        texts = load_perplexity_corpus(proc_dir, source=ppl_src) if ppl_src else []
        if texts:
            if hf_model is None:
                hf_model_, hf_tok_ = load_hf_model_and_tokenizer(
                    model_id_or_path=var["model"],
                    precision=precision,
                    attn_impl=attn_impl,
                    compile_model=False,
                    grad_ckpt=False,
                    allow_tf32=True,
                    max_model_len=int(cfg.get("hf", {}).get("max_model_len", 8192)),
                    tokenizer_dir_override=var.get("tokenizer"),
                    device_map=cfg.get("hf", {}).get("device_map", "auto"),
                )
                hf_model = hf_model_ ; hf_tok = hf_tok_
            device = next(hf_model.parameters()).device
            hf_model.eval()
            nlls = []
            ppl_bs = int(cfg.get("hf", {}).get("ppl_batch_size", 8))
            with torch.no_grad():
                for i in range(0, len(texts), ppl_bs):
                    batch = texts[i:i+ppl_bs]
                    enc = hf_tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                    out = hf_model(**enc)
                    logits = out.logits[:, :-1, :]
                    labels = enc.input_ids[:, 1:]
                    if hf_tok.pad_token_id is None:
                        mask = torch.ones_like(labels, dtype=torch.bool)
                    else:
                        mask = labels != hf_tok.pad_token_id
                    vocab = logits.size(-1)
                    loss_per_token = F.cross_entropy(
                        logits.reshape(-1, vocab),
                        labels.reshape(-1),
                        reduction="none",
                    ).view(labels.size())
                    masked = loss_per_token * mask.float()
                    counts = mask.sum(dim=1).clamp(min=1)
                    per_sample = masked.sum(dim=1) / counts
                    nlls.extend(float(v) for v in per_sample.detach().cpu().tolist() if not math.isnan(v))
            metrics["perplexity"] = float(perplexity(nlls))
            samples.append(f"PPL: {metrics['perplexity']:.2f}")

        # Optional tokenizer analysis (base vs adapted)
        try:
            if bool(cfg.get("enable_tokenizer_analysis", False)):
                from .tokenizer_analysis import analyze as _analyze_tok
                # Attempt to locate added_terms from latest adapted artifacts (best-effort)
                added_terms = None
                if adapted and adapted.tokenizer_dir:
                    cand = os.path.join(os.path.dirname(os.path.dirname(adapted.tokenizer_dir)), "added_terms.txt")
                    if os.path.isfile(cand):
                        added_terms = cand
                analysis = _analyze_tok(
                    base_tokenizer_path=var["model"],
                    adapted_tokenizer_path=(var.get("tokenizer") or None),
                    bench_dir=bench_dir,
                    proc_dir=proc_dir,
                    biomed_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "biomed_corpus"),
                    added_terms_path=added_terms,
                )
                with open(os.path.join(var_dir, "tokenizer_analysis.json"), "w", encoding="utf-8") as fa:
                    json.dump(analysis, fa, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[MedTokAlign] Tokenizer analysis skipped: {e}")

        # Save outputs
        save_json(metrics, os.path.join(var_dir, "metrics.json"))
        for name, rows in per_dataset_outputs.items():
            save_json(rows, os.path.join(var_dir, f"{name}.json"))
        with open(os.path.join(var_dir, "samples.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(samples))
        print(json.dumps({var["name"]: metrics}, indent=2))
        summary[var["name"]] = metrics
        variant_correct[var["name"]] = per_dataset_correct

    if summary:
        if len(summary) > 1:
            save_json(summary, os.path.join(run_dir, "metrics_summary.json"))
            # Also emit compact CSV
            rows: List[Tuple[str, str, str, float]] = []
            for var_name, mets in summary.items():
                for k, v in mets.items():
                    rows.append((var_name, k.replace("_acc", ""), "acc" if k.endswith("_acc") else k, float(v)))
            save_csv(rows, os.path.join(run_dir, "results.csv"))
            # Optional statistical significance
            try:
                if bool(cfg.get("enable_stats", False)) and len(variant_correct) >= 2:
                    from .eval_stats import bootstrap_diff, save_stats
                    # choose two variants if more than two
                    names = list(variant_correct.keys())
                    base_name = "base" if "base" in names else names[0]
                    adapted_name = "adapted" if "adapted" in names else names[1 if names[1] != base_name else 0]
                    base_map = variant_correct.get(base_name, {})
                    adapt_map = variant_correct.get(adapted_name, {})
                    datasets = sorted(set(base_map.keys()) & set(adapt_map.keys()))
                    stats = {}
                    for ds in datasets:
                        b = base_map.get(ds) or []
                        a = adapt_map.get(ds) or []
                        if b and a and len(b) == len(a):
                            stats[ds] = bootstrap_diff([float(x) for x in b], [float(x) for x in a], n_samples=1000, seed=int(cfg.get("random_seed", 17)))
                    if stats:
                        save_stats(run_dir, stats)
            except Exception as e:
                print(f"[MedTokAlign] Stats computation skipped: {e}")
        return

    # (Legacy single-variant path removed; handled above.)


if __name__ == "__main__":
    main()


