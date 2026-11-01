"""Convert a llama.cpp Qwen3 GGUF (e.g. Utter's cleanup model) into an MLX checkpoint.

Dequantizes every tensor, maps llama.cpp names to Hugging Face Qwen3 names,
takes config and tokenizer files from the base model, and re-quantizes with MLX.

Usage: uv run python convert_gguf.py <model.gguf> <out-dir> [--base Qwen/Qwen3-1.7B] [--bits 6]
"""

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from mlx_lm import convert

LAYER_NAMES = {
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
GLOBAL_NAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
BASE_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]


def hf_name(name: str) -> str:
    if name in GLOBAL_NAMES:
        return GLOBAL_NAMES[name]
    match = re.fullmatch(r"blk\.(\d+)\.(\w+)\.(weight|bias)", name)
    if not match or match.group(2) not in LAYER_NAMES:
        raise KeyError(f"unmapped GGUF tensor {name}")
    layer, part, kind = match.groups()
    return f"model.layers.{layer}.{LAYER_NAMES[part]}.{kind}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument(
        "--base-revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    reader = GGUFReader(args.gguf)
    if reader.fields["general.architecture"].contents() != "qwen3":
        raise SystemExit("only qwen3 GGUFs are supported")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        weights = {}
        for tensor in reader.tensors:
            values = dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
            weights[hf_name(tensor.name)] = mx.array(values).astype(mx.bfloat16)
        mx.save_safetensors(
            str(staging / "model.safetensors"), weights, metadata={"format": "pt"}
        )

        for name in BASE_FILES:
            try:
                path = hf_hub_download(args.base, name, revision=args.base_revision)
            except EntryNotFoundError:
                print(f"skipping {name}: not in {args.base}")
                continue
            shutil.copy(path, staging / name)
        config = json.loads((staging / "config.json").read_text())
        config["tie_word_embeddings"] = "lm_head.weight" not in weights
        (staging / "config.json").write_text(json.dumps(config, indent=2))
        template = reader.fields.get("tokenizer.chat_template")
        if template is not None:
            (staging / "chat_template.jinja").write_text(template.contents())

        out = Path(args.out)
        if out.exists():
            shutil.rmtree(out)
        convert(
            str(staging),
            mlx_path=str(out),
            quantize=True,
            q_bits=args.bits,
            q_group_size=args.group_size,
        )
        if (staging / "chat_template.jinja").exists():
            shutil.copy(staging / "chat_template.jinja", out / "chat_template.jinja")
    print(f"wrote {out} ({len(weights)} tensors, {args.bits}-bit)")


if __name__ == "__main__":
    main()
