import argparse
import itertools
import math
import os
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_texts(bench_dir: str, proc_dir: str, max_docs: int = 20000) -> List[str]:
    def load_jsonl(path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        yield __import__("json").loads(line)
                    except Exception:
                        continue
        except FileNotFoundError:
            return []

    # Only use PubMedQA and MedQA
    files = [
        "pubmedqa_train.jsonl","pubmedqa_validation.jsonl","pubmedqa_test.jsonl",
        "medqa_train.jsonl","medqa_validation.jsonl","medqa_test.jsonl",
    ]
    texts: List[str] = []
    for name in files:
        for ex in load_jsonl(os.path.join(bench_dir, name)):
            for k in ("text","document","passages","context","long_answer","premise","hypothesis","question"):
                v = ex.get(k)
                if isinstance(v, list):
                    v = "\n".join(map(str, v))
                if isinstance(v, str) and v:
                    texts.append(v)
                    if len(texts) >= max_docs:
                        return texts
    for ex in load_jsonl(os.path.join(proc_dir, "pubmed_rct_test.jsonl")):
        v = ex.get("text") or ex.get("abstract") or ex.get("sentence")
        if isinstance(v, list):
            v = "\n".join(map(str, v))
        if isinstance(v, str) and v:
            texts.append(v)
            if len(texts) >= max_docs:
                break
    return texts


def warmup_rows(
    adapted_model_path: str,
    base_model_id: str,
    tokenizer_dir: str,
    bench_dir: str,
    proc_dir: str,
    steps: int = 1000,
    lr: float = 5e-4,
    per_device_batch_size: int = 4,
    max_len: int = 1024,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype)
    base_vocab = base_model.get_input_embeddings().weight.shape[0]
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(adapted_model_path, torch_dtype=dtype)
    model.train()
    model.to(device)

    embed = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings().weight if model.get_output_embeddings() is not None else None
    assert lm_head is not None, "Model must have tied or explicit lm_head"

    for p in model.parameters():
        p.requires_grad = False
    embed.requires_grad_(True)
    lm_head.requires_grad_(True)

    opt = torch.optim.AdamW([embed, lm_head], lr=lr)

    texts = load_texts(bench_dir, proc_dir)
    def batches() -> Iterable[List[str]]:
        it = itertools.cycle(texts) if texts else iter(())
        while True:
            batch = []
            for _ in range(per_device_batch_size):
                try:
                    t = next(it)
                except StopIteration:
                    return
                batch.append(t)
            yield batch

    it = batches()
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            break
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, labels=enc["input_ids"])
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Zero out gradients for old rows
        if embed.grad is not None and embed.grad.shape[0] > base_vocab:
            embed.grad[:base_vocab].zero_()
        if lm_head.grad is not None and lm_head.grad.shape[0] > base_vocab:
            lm_head.grad[:base_vocab].zero_()
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"[Warmup] step={step+1}/{steps} loss={loss.item():.4f}")

    # Save in-place
    model.save_pretrained(adapted_model_path)


def main():
    ap = argparse.ArgumentParser(description="Warm up new embedding rows only")
    ap.add_argument("--adapted_model_path", type=str, required=True)
    ap.add_argument("--base_model_id", type=str, required=True)
    ap.add_argument("--tokenizer_dir", type=str, required=True)
    ap.add_argument("--bench_dir", type=str, required=True)
    ap.add_argument("--proc_dir", type=str, required=True)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--per_device_batch_size", type=int, default=4)
    args = ap.parse_args()

    if args.warmup_steps and args.warmup_steps > 0:
        warmup_rows(
            adapted_model_path=args.adapted_model_path,
            base_model_id=args.base_model_id,
            tokenizer_dir=args.tokenizer_dir,
            bench_dir=args.bench_dir,
            proc_dir=args.proc_dir,
            steps=int(args.warmup_steps),
            lr=float(args.lr),
            per_device_batch_size=int(args.per_device_batch_size),
        )


if __name__ == "__main__":
    main()


