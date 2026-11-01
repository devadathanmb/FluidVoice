"""Score runs on real dictation history: term checks, drift, and latency.

Usage: uv run python score_history.py results/hist-*.json [--show TAG] [--terms TAG]
"""

import argparse
import difflib
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).parent


def load_checks(path):
    return {
        c["id"]: c
        for c in map(json.loads, Path(path).read_text().splitlines())
        if c.get("must") or c.get("must_not")
    }


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def drift(raw, out):
    """Similarity to the raw transcript and length ratio. Cleanup should stay close to both."""
    a, b = words(raw), words(out)
    sim = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a else 1.0
    ratio = len(b) / max(1, len(a))
    return sim, ratio


def is_flagged(r):
    sim, ratio = drift(r["input"], r["output"])
    if len(words(r["input"])) <= 3:
        return sim < 0.5
    return sim < 0.6 or ratio < 0.55 or ratio > 1.3 or r.get("finish") == "length"


def term_results(r, check):
    out = r["output"]
    fails = [
        f"missing {p}"
        for p in check.get("must", [])
        if not re.search(p, out, re.MULTILINE)
    ]
    fails += [
        f"has {p}" for p in check.get("must_not", []) if re.search(p, out, re.MULTILINE)
    ]
    total = len(check.get("must", [])) + len(check.get("must_not", []))
    return total - len(fails), total, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--cases", default=str(HERE / "private" / "history_cases.jsonl"))
    ap.add_argument("--show", help="print flagged cases for runs containing this tag")
    ap.add_argument(
        "--terms", help="print term-check failures for runs containing this tag"
    )
    args = ap.parse_args()
    checks = load_checks(args.cases)

    hdr = f"{'run':<40} {'terms':>7} {'casesOK':>7} {'flag':>4} {'sim':>5} {'p50ms':>6} {'p90ms':>6} {'maxms':>6} {'>10s':>4} {'tok/s':>5}"
    print(hdr + "\n" + "-" * len(hdr))
    for f in args.files:
        res = json.loads(Path(f).read_text())["results"]
        passed = total = ok_cases = 0
        for r in res:
            if r["id"] in checks:
                p, t, fails = term_results(r, checks[r["id"]])
                passed, total, ok_cases = passed + p, total + t, ok_cases + (not fails)
        flags = [r for r in res if is_flagged(r)]
        sims = [drift(r["input"], r["output"])[0] for r in res]
        lat = sorted(r["total_ms"] for r in res)
        n = len(lat)
        tps = statistics.median(r["gen_tps"] for r in res if r["gen_tokens"] > 8)
        name = Path(f).stem
        print(
            f"{name:<40} {passed / max(1, total):>7.0%} {ok_cases:>3}/{len(checks):<3} {len(flags):>4} {statistics.mean(sims):>5.2f} "
            f"{lat[n // 2]:>6.0f} {lat[int(n * 0.9)]:>6.0f} {lat[-1]:>6.0f} {sum(x > 10000 for x in lat):>4} {tps:>5.0f}"
        )
        if args.terms and args.terms in name:
            for r in res:
                if r["id"] in checks:
                    _, _, fails = term_results(r, checks[r["id"]])
                    if fails:
                        print(f"   [{r['id']}] {'; '.join(fails)}")
        if args.show and args.show in name:
            for r in flags:
                sim, ratio = drift(r["input"], r["output"])
                print(
                    f"\n   [{r['id']}] sim={sim:.2f} len={ratio:.2f} finish={r.get('finish')}"
                )
                print("     IN :", r["input"][:300].replace("\n", "\\n"))
                print("     OUT:", r["output"][:300].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
