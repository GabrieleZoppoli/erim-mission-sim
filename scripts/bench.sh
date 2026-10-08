#!/usr/bin/env bash
# Environment report + smoke tests + benchmark. Run first on the compute node; takes about 5 minutes.
# usage: bash scripts/bench.sh <out_dir> [gpu]
set -euo pipefail
OUT=${1:?out_dir}; GPU=${2:-0}
mkdir -p "$OUT"
{
  echo "== $(date -u)"; nvidia-smi || true; python3 --version; pip show jax jaxlib 2>/dev/null | grep -E "^(Name|Version)" || true
  echo "== smoke tests (CPU-sized, on the selected GPU)"
  for E in E1 E2 E3 E4; do python3 -m erimsim --exp $E --out "$OUT/smoke" --smoke --gpu "$GPU" | tail -2; done
  echo "== benchmark at full size"
  python3 -m erimsim.bench --gpu "$GPU"
} 2>&1 | tee "$OUT/bench_report.txt"
echo "report: $OUT/bench_report.txt"
