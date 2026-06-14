#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

batch_size=2048
numsteps=3000

dataset_path="data/simulated/balls_4way/transforms_in_in.json"
/bin/bash "$PROJECT_ROOT/scripts/run_canonical.sh" "$dataset_path" "canonical" "$batch_size" "in_in" "$numsteps"
/bin/bash "$PROJECT_ROOT/scripts/run_canonical.sh" "$dataset_path" "eval" "$batch_size" "nerf_xray/canonical_in_in" "$numsteps"

dataset_path="data/simulated/balls_4way/transforms_out_out.json"
/bin/bash "$PROJECT_ROOT/scripts/run_canonical.sh" "$dataset_path" "canonical" "$batch_size" "out_out" "$numsteps"
/bin/bash "$PROJECT_ROOT/scripts/run_canonical.sh" "$dataset_path" "eval" "$batch_size" "nerf_xray/canonical_out_out" "$numsteps"

dataset_path="data/simulated/balls_4way/transforms_out_in.json"
/bin/bash "$PROJECT_ROOT/scripts/run_canonical.sh" "$dataset_path" "canonical" "$batch_size" "out_in" "$numsteps"
/bin/bash "$PROJECT_ROOT/scripts/run_canonical.sh" "$dataset_path" "eval" "$batch_size" "nerf_xray/canonical_out_in" "$numsteps"

dataset_path="data/simulated/balls_4way/transforms_in_out.json"
/bin/bash "$PROJECT_ROOT/scripts/run_canonical.sh" "$dataset_path" "canonical" "$batch_size" "in_out" "$numsteps"
/bin/bash "$PROJECT_ROOT/scripts/run_canonical.sh" "$dataset_path" "eval" "$batch_size" "nerf_xray/canonical_in_out" "$numsteps"