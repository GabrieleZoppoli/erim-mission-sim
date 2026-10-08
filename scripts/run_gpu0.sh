#!/usr/bin/env bash
# GPU 0 stream: E1 (main comparison) -> E3 (dual effect, warm start from E1) -> E2 CNN sweep + reference.
# usage: nohup bash scripts/run_gpu0.sh <out_dir> > <out_dir>/gpu0.log 2>&1 &
set -uo pipefail
OUT=${1:?out_dir}; mkdir -p "$OUT"
python3 -m erimsim --exp E1 --out "$OUT" --gpu 0 --resume
python3 -m erimsim --exp E3 --out "$OUT" --gpu 0 --resume
python3 -m erimsim --exp E2 --out "$OUT" --gpu 0 --resume --units 'cnn_*'
python3 -m erimsim --exp E2 --out "$OUT" --gpu 0 --resume --units 'ref'
echo "GPU0 stream finished $(date -u)"
