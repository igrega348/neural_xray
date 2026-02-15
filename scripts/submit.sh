#!/bin/bash

###############################################################################
# submit.sh - Submit a complete training pipeline for neural X-ray reconstruction
#
# This script runs a full training pipeline consisting of:
#   1. Canonical volume training (forward and backward)
#   2. Velocity field training at multiple resolutions (6→9→15→27→51 control points)
#   3. Spatiotemporal mixing training
#
# Usage:
#   ./scripts/submit.sh <dataset_path>
#
# Arguments:
#   dataset_path    Path to the dataset directory (relative to project root)
#                   Example: "data/synthetic/balls" or "data/experimental/kel_F"
#
# Example:
#   ./scripts/submit.sh data/synthetic/balls
#
# The script will run multiple training stages sequentially, with each stage
# building on the previous one. The total training involves ~18,000 steps.
###############################################################################

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNSCRIPT_PATH="$SCRIPT_DIR/run_dset.sh"

# Check if dataset_path argument is provided
if [ -z "$1" ]; then
    echo "Error: dataset_path argument is required" >&2
    echo ""
    echo "Usage: $0 <dataset_path>"
    echo ""
    echo "Arguments:"
    echo "  dataset_path    Path to the dataset directory (relative to project root)"
    echo "                  Example: 'data/synthetic/balls' or 'data/experimental/kel_F'"
    echo ""
    echo "Example:"
    echo "  $0 data/synthetic/balls"
    exit 1
fi

dataset_path=$1

numsteps=3000
/bin/bash "$RUNSCRIPT_PATH" "$dataset_path" canonical 2048 '' '' '' '' '' '' '' '' $numsteps || exit 1
steps=$numsteps # 3000
/bin/bash "$RUNSCRIPT_PATH" "$dataset_path" vfield 2048 '' '' 6 6 $steps 1e-3 1000 0.1 $numsteps || exit 1
steps=$(($steps + $numsteps)) # 6000
/bin/bash "$RUNSCRIPT_PATH" "$dataset_path" vfield 1024 '' '' 6 9 $steps 1e-3 1000 0.09 $numsteps || exit 1
steps=$(($steps + $numsteps)) # 9000
/bin/bash "$RUNSCRIPT_PATH" "$dataset_path" vfield 512 '' '' 9 15 $steps 1e-7 200 0.085 $numsteps || exit 1
steps=$(($steps + $numsteps)) # 12000
/bin/bash "$RUNSCRIPT_PATH" "$dataset_path" vfield 1024 '' '' 15 27 $steps 1e-7 200 0.075 $numsteps || exit 1	
steps=$(($steps + $numsteps)) # 15000
/bin/bash "$RUNSCRIPT_PATH" "$dataset_path" vfield 1024 '' '_2' 27 51 $steps 1e-7 200 0.0499 $numsteps || exit 1	
steps=$(($steps + $numsteps)) # 18000
/bin/bash "$RUNSCRIPT_PATH" "$dataset_path" spatiotemporal_mix 2048 '' '' 51 9 $steps 1e-7 200 0.0499 $numsteps || exit 1
# /bin/bash "$RUNSCRIPT_PATH" "$dataset_path" export 500 'spatiotemporal_mix/vel_51' 'mixed' || exit 1
# /bin/bash "$RUNSCRIPT_PATH" "$dataset_path" export 500 'spatiotemporal_mix/vel_51' 'forward' || exit 1
# /bin/bash "$RUNSCRIPT_PATH" "$dataset_path" export 500 'spatiotemporal_mix/vel_51' 'backward' || exit 1
# /bin/bash "$RUNSCRIPT_PATH" "$dataset_path" eval 0 'spatiotemporal_mix/vel_51_9' || exit 1
