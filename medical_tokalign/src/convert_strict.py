import argparse
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


_EMBED_DICT = {
    "gpt_neox": "gpt_neox.embed_in.weight",
    "llama": "model.embed_tokens.weight",
    "mistral": "model.embed_tokens.weight",
    "qwen2": "model.embed_tokens.weight",
    "qwen2_moe": "model.embed_tokens.weight",
}

_LMHEAD_DICT = {
    "gpt_neox": "embed_out.weight",
    "llama": "lm_head.weight",
    "mistral": "lm_head.weight",
    "qwen2": "lm_head.weight",
    "qwen2_moe": "lm_head.weight",
}


def trans2switch_strict(
    trans_path: str,
    src_clm_path: str,
    tgt_clm_path: str,
    tgt_tok_path: str,
    random_shuffle: float = -1.0,
    trust_remote_code: bool = False,
):
    src_model = AutoModelForCausalLM.from_pretrained(
        src_clm_path, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code
    )
    tgt_tok = AutoTokenizer.from_pretrained(tgt_tok_path, trust_remote_code=trust_remote_code)

    with open(trans_path, "r") as f:
        trans = json.load(f)

    state = src_model.state_dict()
    embed_key = _EMBED_DICT[src_model.config.model_type]
    lm_head_key = _LMHEAD_DICT[src_model.config.model_type]
    src_embed = state[embed_key]
    src_lm_head = state[lm_head_key]

    assert src_embed.shape[0] == src_lm_head.shape[0]
    hid_dim = src_embed.shape[1]
    src_len = src_embed.shape[0]
    tgt_len = len(list(trans.keys()))

    tgt_embed = torch.zeros((tgt_len, hid_dim), dtype=src_embed.dtype, device=src_embed.device)
    tgt_lm_head = torch.zeros((tgt_len, hid_dim), dtype=src_lm_head.dtype, device=src_lm_head.device)

    for i in range(tgt_len):
        tj = trans[f"{i}"]
        if random_shuffle > 0 and random.random() < random_shuffle:
            tj = random.randint(0, src_len - 1)
        tgt_embed[i] = src_embed[tj]
        tgt_lm_head[i] = src_lm_head[tj]

    # Ensure shapes match by resizing embeddings first
    src_model.resize_token_embeddings(tgt_len)
    # Refresh state dict after resize so strict=True validates updated tensor shapes
    state = src_model.state_dict()
    state[embed_key] = tgt_embed.to(torch.bfloat16)
    state[lm_head_key] = tgt_lm_head.to(torch.bfloat16)
    src_model.load_state_dict(state, strict=True)

    src_model.save_pretrained(tgt_clm_path)
    tgt_tok.save_pretrained(tgt_clm_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--one2one-matrix-path", type=str, required=True)
    parser.add_argument("-s", "--source-model-path", type=str, required=True)
    parser.add_argument("-t", "--target-tokenizer-path", type=str, required=True)
    parser.add_argument("-o", "--output-model-path", type=str, required=True)
    parser.add_argument(
        "-r",
        "--random-shuffle-percentage",
        type=float,
        default=-1,
        help="The percentage of token pairs that are randomly shuffled rather than map to the target.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of custom model code from model repos. Default: False",
    )
    args = parser.parse_args()

    trans2switch_strict(
        trans_path=args.one2one_matrix_path,
        src_clm_path=args.source_model_path,
        tgt_clm_path=args.output_model_path,
        tgt_tok_path=args.target_tokenizer_path,
        random_shuffle=args.random_shuffle_percentage,
        trust_remote_code=bool(args.trust_remote_code),
    )


if __name__ == "__main__":
    main()


