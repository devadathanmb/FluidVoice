"""Score benchmark result files.

Usage: uv run python score.py results/*.json [--fails TAG]
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def norm_exact(s: str) -> str:
    s = s.translate(QUOTES).strip()
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def words(s: str) -> list[str]:
    s = s.translate(QUOTES).lower()
    s = re.sub(
        r"(?m)^\s*(\d+\.|-)\s+", " ", s
    )  # list markers are structure, not content
    s = re.sub(r"[^\w\s'$%:./-]", " ", s)
    s = re.sub(r"(?<!\d)[.:/-]|[.:/-](?!\d)", " ", s)
    return s.split()


def wer(hyp: str, ref: str) -> float:
    h, r = words(hyp), words(ref)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


def structure(s: str) -> tuple:
    kinds = []
    for ln in norm_exact(s).split("\n"):
        if not ln:
            kinds.append("blank")
        elif re.match(r"^\d+\.\s", ln):
            kinds.append("num")
        elif re.match(r"^[-*•]\s", ln):
            kinds.append("bullet")
        else:
            kinds.append("text")
    return tuple(kinds)


def score_case(r):
    out = r["output"]
    refs = r["refs"]
    exact = any(norm_exact(out) == norm_exact(ref) for ref in refs)
    w = min(wer(out, ref) for ref in refs)
    struct = any(structure(out) == structure(ref) for ref in refs)
    return exact, w, struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument(
        "--fails",
        help="print non-exact cases for result files whose name contains this",
    )
    ap.add_argument("--wer-threshold", type=float, default=0.0)
    args = ap.parse_args()

    header = f"{'run':<44} {'exact':>6} {'struct':>6} {'WER':>6} {'bad':>4} {'p50ms':>6} {'p90ms':>6} {'tok/s':>6} {'GB':>5}"
    print(header)
    print("-" * len(header))
    rows = []
    for f in args.files:
        data = json.loads(Path(f).read_text())
        res = data["results"]
        sc = [score_case(r) for r in res]
        n = len(res)
        exact = sum(e for e, _, _ in sc) / n
        struct = sum(s for _, _, s in sc) / n
        mw = statistics.mean(w for _, w, _ in sc)
        bad = sum(w > 0.4 for _, w, _ in sc)
        lat = sorted(r["total_ms"] for r in res)
        tps = statistics.median(r["gen_tps"] for r in res if r["gen_tokens"] > 4)
        peak = max(r["peak_gb"] for r in res)
        name = Path(f).stem
        rows.append((name, res, sc))
        print(
            f"{name:<44} {exact:>6.0%} {struct:>6.0%} {mw:>6.3f} {bad:>4} {lat[n // 2]:>6.0f} {lat[int(n * 0.9)]:>6.0f} {tps:>6.0f} {peak:>5.1f}"
        )

    print("\nper-category mean WER")
    cats = sorted({r["cat"] for _, res, _ in rows for r in res})
    print(f"{'run':<44} " + " ".join(f"{c[:9]:>9}" for c in cats))
    for name, res, sc in rows:
        by = defaultdict(list)
        for r, (_, w, _) in zip(res, sc):
            by[r["cat"]].append(w)
        print(
            f"{name:<44} " + " ".join(f"{statistics.mean(by[c]):>9.3f}" for c in cats)
        )

    if args.fails:
        for name, res, sc in rows:
            if args.fails not in name:
                continue
            print(f"\n===== non-exact cases: {name}")
            for r, (e, w, s) in zip(res, sc):
                if e or w < args.wer_threshold:
                    continue
                print(f"\n[{r['id']}] WER={w:.2f} struct={'ok' if s else 'NO'}")
                print("  IN :", r["input"].replace("\n", "\\n"))
                print("  OUT:", r["output"].replace("\n", "\\n"))
                print("  REF:", r["refs"][0].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
