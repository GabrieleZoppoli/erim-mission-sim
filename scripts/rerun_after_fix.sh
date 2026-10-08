#!/usr/bin/env bash
# After the policy-search fix of 8 Oct 2026: set aside the void ERIM results and their markers so that --resume
# reruns them, and keep what is still valid (E2 cnn_* units, E4). Run from the NEW clone, against the same --out.
# usage: bash scripts/rerun_after_fix.sh <out_dir>
set -uo pipefail
OUT=${1:?out_dir}; TS=$(date -u +%Y%m%d_%H%M)
if [ -d "$OUT/E1" ]; then mv "$OUT/E1" "$OUT/E1_void_$TS"; echo "E1 set aside as E1_void_$TS (baseline rows are regenerated, 3 minutes)"; fi
if [ -d "$OUT/E2" ]; then
  for f in rates_est.csv rates_ref.csv; do [ -f "$OUT/E2/$f" ] && mv "$OUT/E2/$f" "$OUT/E2/${f%.csv}_void_$TS.csv"; done
  rm -f "$OUT/E2/units/est_"*.done "$OUT/E2/units/ref.done"
  echo "E2: est_* and ref set aside and unmarked; cnn_* kept"
fi
if [ -d "$OUT/E3" ]; then mv "$OUT/E3" "$OUT/E3_void_$TS"; echo "E3 set aside"; fi
echo "Then, from this clone:"
echo "  nohup bash scripts/run_gpu0.sh $OUT > $OUT/gpu0_rerun.log 2>&1 &    # E1 -> E3 -> E2 cnn_* (skipped if done) -> ref"
echo "  nohup bash scripts/run_gpu1.sh $OUT > $OUT/gpu1_rerun.log 2>&1 &    # E2 est_* two shards -> E4 (skipped if done)"
