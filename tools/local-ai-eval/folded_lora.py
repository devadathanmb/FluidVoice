"""Apply a PEFT-orientation LoRA whose alpha/r scale is folded into B.

Glimpse C1 ships `lora_A` as [r, in] and `lora_B` as [out, r] keyed by
`model.layers.N.<module>`; the adapter sees the layer's unrotated input.
"""

from pathlib import Path

import mlx.core as mx
from mlx import nn


class FoldedLoRA(nn.Module):
    def __init__(self, base: nn.Module, a: mx.array, b: mx.array):
        super().__init__()
        self.base = base
        self.a_t = a.T
        self.b_t = b.T

    def __call__(self, x):
        return self.base(x) + (
            (x @ self.a_t.astype(x.dtype)) @ self.b_t.astype(x.dtype)
        )


def apply(model: nn.Module, adapter_path: Path) -> int:
    weights = mx.load(str(adapter_path))
    targets = {k.rsplit(".lora_", 1)[0] for k in weights}
    modules = dict(model.named_modules())
    applied = 0
    for target in sorted(targets):
        path = next(
            (p for p in modules if p == target or p.endswith("." + target)), None
        )
        if path is None:
            raise KeyError(f"adapter target {target} not found in model")
        parent_path, _, name = path.rpartition(".")
        parent = modules[parent_path] if parent_path else model
        wrapped = FoldedLoRA(
            getattr(parent, name),
            weights[f"{target}.lora_A"],
            weights[f"{target}.lora_B"],
        )
        setattr(parent, name, wrapped)
        applied += 1
    mx.eval(model.parameters())
    return applied
