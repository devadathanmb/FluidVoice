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

- `convert_gguf.py` turns a llama.cpp Qwen3 GGUF into a standard MLX checkpoint.
  It produced the app's published model,
  `devadathanmb/utter-dictation-cleanup-qwen3-1.7b-mlx-6bit`, from Utter's Q6_K GGUF.
- `convert_fused.py` splits Fluid-1's fused `linear_attn.in_proj` tensor into the
  stock Qwen3.5 layout so mlx-lm can load it (evaluation only).
- `folded_lora.py` applies Glimpse C1's adapter (PEFT layout, scale folded into B).
- `bench.py --server URL` benchmarks a llama-server endpoint instead of MLX.
- `private/` holds personal data: dictation history, glossary, and prompts.
  `prompts/` holds prompt profiles copied from a local FluidVoice.app install.
  Both are git-ignored, as are history results.

Findings on 145 real technical dictations (M4 Pro, 24 GB), technical-term checks
and outputs that drifted from what was said:

| Model | Terms | Drift | Median latency |
|---|---|---|---|
| Gemini Flash Lite (cloud baseline) | 84% | 4 | n/a |
| Utter Qwen3-1.7B + vocabulary (llama.cpp) | 62% | 0 | 598 ms |
| Utter Qwen3-1.7B + vocabulary (MLX 6-bit, shipped) | 58% | 0 | 745 ms |
| Fluid-1 Mini + vocabulary | 43% | 5 | 844 ms |
| Glimpse C1 2B | 19% | 0 | 668 ms |
| Qwen3.5-4B base + vocabulary | 78% | 19 | 1330 ms |

Base models repair more terms but rewrite long dictation. Remaining gaps are
context homophones (bot/board, reCAPTCHA/recapture), the target for fine-tuning.
