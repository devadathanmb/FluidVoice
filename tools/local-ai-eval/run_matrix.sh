#!/bin/bash
# Usage: ./run_matrix.sh "<model>|<tag>" ... -- variant ...  (history and synthetic sets)
set -u
cd "$(dirname "$0")"
models=(); variants=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do models+=("$1"); shift; done
shift
variants=("$@")
for m in "${models[@]}"; do
  src=${m%%|*}; tag=${m##*|}
  for v in "${variants[@]}"; do
    uv run python bench.py "$src" --variant "$v" --cases private/history_cases.jsonl --tag "hist-$tag-$v" > "/tmp/fvhf/hist-$tag-$v.log" 2>&1 || echo "FAIL hist $tag $v"
    uv run python bench.py "$src" --variant "$v" --tag "$tag-$v" > "/tmp/fvhf/syn-$tag-$v.log" 2>&1 || echo "FAIL syn $tag $v"
    echo "done $tag $v"
  done
done
echo MATRIX_DONE
