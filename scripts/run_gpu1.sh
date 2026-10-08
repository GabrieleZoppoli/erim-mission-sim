#!/usr/bin/env bash
# GPU 1 stream: E2 estimator/policy sweep as two concurrent shards (45% of the memory each) -> E4 (PPO).
# usage: nohup bash scripts/run_gpu1.sh <out_dir> > <out_dir>/gpu1.log 2>&1 &
set -uo pipefail
OUT=${1:?out_dir}; mkdir -p "$OUT"
python3 -m erimsim --exp E2 --out "$OUT" --gpu 1 --resume --units 'est_*' --shard 0/2 --mem-fraction 0.45 > "$OUT/e2_shard0.log" 2>&1 &
python3 -m erimsim --exp E2 --out "$OUT" --gpu 1 --resume --units 'est_*' --shard 1/2 --mem-fraction 0.45 > "$OUT/e2_shard1.log" 2>&1 &
wait
python3 -m erimsim --exp E4 --out "$OUT" --gpu 1 --resume
echo "GPU1 stream finished $(date -u)"
