# Local AI cleanup eval

Benchmarks on-device dictation cleanup models with MLX before they ship in
`Sources/Fluid/Services/LocalAI`. The Swift prompt (`LocalCleanupPrompt`) mirrors
the `tech_vocab` variant in `prompts.py`; change both together.

```sh
uv sync
# Synthetic cases (cases.jsonl): quality, latency, memory
uv run python bench.py altic-dev/fluid-1-mini --variant sys   # see convert_fused.py first
uv run python score.py results/*.json
# Real dictation history (private/, never committed) plus synthetic cases
./run_matrix.sh "<hf-repo-or-dir>|<tag>" -- tech_vocab
uv run python score_history.py results/hist-*.json --terms <tag> --show <tag>
```

- `convert_fused.py` splits Fluid-1's fused `linear_attn.in_proj` tensor into the
  stock Qwen3.5 layout so mlx-lm and mlx-swift-lm can load it. The app does the
  same split after download.
- `private/` holds personal data: dictation history, glossary, and prompts.
  `prompts/` holds prompt profiles copied from a local FluidVoice.app install.
  Both are git-ignored, as are history results.

Findings (M4 Pro, 24 GB): Fluid-1 Mini is the fastest faithful model (about 160 ms
median, about 2 GB). Base Qwen3.5-4B recognizes more technical terms but rewrites
long dictation. Neither matches a cloud model on technical-term repair without a
vocabulary or fine-tune.
