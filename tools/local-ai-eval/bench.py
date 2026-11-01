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

import folded_lora
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
    ap.add_argument(
        "--server", help="OpenAI-compatible base URL (llama-server); skips MLX loading"
    )
    ap.add_argument(
        "--folded-adapter", help="PEFT LoRA with scale folded into B (Glimpse C1)"
    )
    args = ap.parse_args()

    path = Path(args.model)
    if not args.server and not path.exists():
        path = Path(
            snapshot_download(
                args.model, allow_patterns=[f"{args.subdir}/*"] if args.subdir else None
            )
        )
    if args.subdir:
        path = path / args.subdir

    if args.server:
        return run_server(args)

    t0 = time.perf_counter()
    model, tokenizer = load(str(path))
    if args.folded_adapter:
        folded_lora.apply(model, Path(args.folded_adapter))
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


def run_server(args):
    """Benchmark a llama-server endpoint with the same cases and result format."""
    import urllib.request

    cases = [
        json.loads(line)
        for line in Path(args.cases).read_text().splitlines()
        if line.strip()
    ]
    results = []
    for case in cases:
        body = {
            "messages": prompts.messages(args.variant, case["input"]),
            "temperature": 0,
            "max_tokens": len(case["input"]) // 2 + 64,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        start = time.perf_counter()
        request = urllib.request.Request(
            args.server.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read())
        end = time.perf_counter()
        out = THINK_RE.sub("", data["choices"][0]["message"]["content"] or "").strip()
        timings = data.get("timings", {})
        results.append(
            {
                "id": case["id"],
                "cat": case["cat"],
                "input": case["input"],
                "refs": case["refs"],
                "output": out,
                "ttft_ms": round(timings.get("prompt_ms", 0), 1),
                "total_ms": round((end - start) * 1000, 1),
                "prompt_tokens": data["usage"]["prompt_tokens"],
                "fed_tokens": timings.get("prompt_n", 0),
                "gen_tokens": data["usage"]["completion_tokens"],
                "gen_tps": round(timings.get("predicted_per_second", 0), 1),
                "peak_gb": 0,
                "finish": data["choices"][0].get("finish_reason"),
            }
        )
        print(
            f"{case['id']:<10} {results[-1]['total_ms']:>7.0f}ms  {out[:90]!r}",
            flush=True,
        )
    out_path = HERE / "results" / f"{args.tag}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"model": args.model, "variant": args.variant, "results": results}, indent=1
        )
    )
    print("wrote", out_path)


if __name__ == "__main__":
    main()
