"""Run dictation-cleanup cases against an MLX model and record quality and speed.

Usage: uv run python bench.py <hf-repo-or-path> --variant fewshot [--subdir mlx-6bit]
"""

import argparse
import copy
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

import prompts

HERE = Path(__file__).parent
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def render(tokenizer, msgs):
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    try:
        return tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, **kwargs)


def build_prefix_cache(model, tokenizer, variant):
    """Prefill the transcript-independent prefix once, as the app will."""
    a = render(tokenizer, prompts.messages(variant, "AAAA"))
    b = render(tokenizer, prompts.messages(variant, "BBBB"))
    common = 0
    while common < min(len(a), len(b)) and a[common] == b[common]:
        common += 1
    prefix_ids = tokenizer.encode(a[:common], add_special_tokens=False)
    # Drop the last token: tokenization at the boundary can merge with the transcript.
    prefix_ids = prefix_ids[:-1]
    if len(prefix_ids) < 8:
        return None, []
    cache = make_prompt_cache(model)
    prompt = mx.array(prefix_ids)
    # Feed in one step; stream_generate with max_tokens=0 is not supported, so run the model directly.
    model(prompt[None], cache=cache)
    mx.eval([c.state for c in cache])
    return cache, prefix_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--variant", default="fewshot")
    ap.add_argument("--subdir", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cases", default=str(HERE / "cases.jsonl"))
    ap.add_argument("--no-prefix-cache", action="store_true")
    args = ap.parse_args()

    path = Path(args.model)
    if not path.exists():
        path = Path(
            snapshot_download(
                args.model, allow_patterns=[f"{args.subdir}/*"] if args.subdir else None
            )
        )
    if args.subdir:
        path = path / args.subdir

    t0 = time.perf_counter()
    model, tokenizer = load(str(path))
    load_s = time.perf_counter() - t0

    cases = [
        json.loads(line)
        for line in Path(args.cases).read_text().splitlines()
        if line.strip()
    ]
    prefix_cache, prefix_ids = (None, [])
    if not args.no_prefix_cache:
        prefix_cache, prefix_ids = build_prefix_cache(model, tokenizer, args.variant)

    # Warm up kernels so the first case's latency is representative.
    for _ in stream_generate(
        model,
        tokenizer,
        render(tokenizer, prompts.messages(args.variant, "hello")),
        max_tokens=4,
    ):
        pass

    results = []
    for case in cases:
        text = render(tokenizer, prompts.messages(args.variant, case["input"]))
        ids = tokenizer.encode(text, add_special_tokens=False)
        cache = None
        feed = ids
        if prefix_cache is not None and ids[: len(prefix_ids)] == prefix_ids:
            cache = copy.deepcopy(prefix_cache)
            feed = ids[len(prefix_ids) :]
        max_tokens = len(tokenizer.encode(case["input"])) * 2 + 48
        start = time.perf_counter()
        first = None
        out = ""
        last = None
        for resp in stream_generate(
            model, tokenizer, mx.array(feed), max_tokens=max_tokens, prompt_cache=cache
        ):
            if first is None:
                first = time.perf_counter()
            out += resp.text
            last = resp
        end = time.perf_counter()
        cleaned = THINK_RE.sub("", out).strip()
        results.append(
            {
                "id": case["id"],
                "cat": case["cat"],
                "input": case["input"],
                "refs": case["refs"],
                "output": cleaned,
                "ttft_ms": round(((first or end) - start) * 1000, 1),
                "total_ms": round((end - start) * 1000, 1),
                "prompt_tokens": len(ids),
                "fed_tokens": len(feed),
                "gen_tokens": last.generation_tokens if last else 0,
                "gen_tps": round(last.generation_tps, 1) if last else 0,
                "peak_gb": round(last.peak_memory, 2) if last else 0,
                "finish": last.finish_reason if last else None,
            }
        )
        print(
            f"{case['id']:<10} {results[-1]['total_ms']:>7.0f}ms  {cleaned[:90]!r}",
            flush=True,
        )

    tag = (
        args.tag
        or f"{Path(args.model).name}{'-' + args.subdir if args.subdir else ''}-{args.variant}"
    )
    out_path = HERE / "results" / f"{tag}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "variant": args.variant,
                "load_s": round(load_s, 2),
                "results": results,
            },
            indent=1,
        )
    )
    print("wrote", out_path)


if __name__ == "__main__":
    main()
