"""Split FluidDecode's fused Qwen3.5 linear-attention projection into the stock layout.

FluidDecode stores `linear_attn.in_proj` as rows [qkv | z | b | a]. Stock MLX
(mlx-lm and mlx-swift-lm) expects separate `in_proj_qkv`, `in_proj_z`,
`in_proj_b`, and `in_proj_a`. Affine quantization groups run along the input
dimension, so slicing rows of weight/scales/biases is lossless.

Usage: uv run python convert_fused.py <src-dir> <dst-dir>
"""

import shutil
import sys
from pathlib import Path

import mlx.core as mx


def split(weights: dict, text_config: dict) -> dict:
    kh, vh = text_config["linear_num_key_heads"], text_config["linear_num_value_heads"]
    kd, vd = text_config["linear_key_head_dim"], text_config["linear_value_head_dim"]
    sizes = {"qkv": 2 * kh * kd + vh * vd, "z": vh * vd, "b": vh, "a": vh}
    out = {}
    for key, value in weights.items():
        marker = "linear_attn.in_proj."
        if marker not in key:
            out[key] = value
            continue
        prefix, suffix = key.split(marker)
        if value.shape[0] != sum(sizes.values()):
            raise ValueError(
                f"{key}: expected {sum(sizes.values())} rows, found {value.shape[0]}"
            )
        offset = 0
        for name, rows in sizes.items():
            out[f"{prefix}linear_attn.in_proj_{name}.{suffix}"] = value[
                offset : offset + rows
            ]
            offset += rows
    return out


def main():
    import json

    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    weights = {}
    for f in sorted(src.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    converted = split(weights, config.get("text_config", config))
    mx.save_safetensors(
        str(dst / "model.safetensors"), converted, metadata={"format": "mlx"}
    )
    for f in src.iterdir():
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            continue
        shutil.copy2(f.resolve(), dst / f.name)
    print(f"wrote {len(converted)} tensors to {dst}")


if __name__ == "__main__":
    main()
