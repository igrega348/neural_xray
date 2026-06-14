#!/bin/bash
# Phase 4 supplement: run compute-normed-correlation for all 4 OOP verification models.
# Expects check_oop.sh to have already trained the models.
# Results saved alongside PSNR JSON files in each model's output directory.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EVAL_PY="$PROJECT_ROOT/nerfstudio/nerfstudio/scripts/eval.py"
OUTDIR="$PROJECT_ROOT/outputs"
DSET="balls_4way"
TARGET_OBJECT="$PROJECT_ROOT/data/simulated/balls_4way/object.json"

if [ ! -f "$TARGET_OBJECT" ]; then
  echo "ERROR: target object not found at $TARGET_OBJECT"
  echo "Run make_balls_4way_linux.sh first."
  exit 1
fi

run_normed() {
  local suf="$1"
  local dname="$OUTDIR/$DSET/$suf"
  local config="$dname/config.yml"
  local out="$dname/eval_normed_${DSET}.json"

  if [ ! -f "$config" ]; then
    echo "SKIP $suf: config not found at $config"
    return
  fi
  if [ -f "$out" ]; then
    echo "SKIP $suf: normed correlation already computed"
    return
  fi
  echo "=== Normed correlation: $suf ==="
  python "$EVAL_PY" compute-normed-correlation \
    --load-config "$config" \
    --target-times 0.0 \
    --target-files "$TARGET_OBJECT" \
    --output-path "$out"
}

run_normed "nerf_xray/canonical_in_in"
run_normed "nerf_xray/canonical_out_out"
run_normed "nerf_xray/canonical_out_in"
run_normed "nerf_xray/canonical_in_out"

echo ""
echo "=== Normed correlation eval done. Run report_oop_matrix.py to see results. ==="
