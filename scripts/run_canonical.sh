#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
pardir="$SCRIPT_DIR"
# run with run_canonical.sh dataset_path mode batch_size suf numsteps
# dataset_path can be either:
#   - A JSON file path (relative or absolute), e.g., "data/experimental/kel_F/transforms_00.json"
#   - A directory path (relative or absolute), e.g., "data/experimental/kel_F" or "data/synthetic/balls"
# If a JSON file is provided, it will be used directly as data0. If a directory is provided, it will search for transforms_[0-9]*.json files.
echo "Running with args: $@"

dataset_path=$1
mode=$2
batch_size=$3
suf=$4
numsteps=$5

# Construct full dataset path
if [[ "$dataset_path" == /* ]]; then
	# Absolute path
	resolved_path="$dataset_path"
else
	# Relative path (relative to project root)
	resolved_path="$PROJECT_ROOT/$dataset_path"
fi

# Check if it's a JSON file or a directory
if [ -f "$resolved_path" ] && [[ "$resolved_path" == *.json ]]; then
	# It's a JSON file - use it directly as data0
	data0="$resolved_path"
	dsetpath="$(dirname "$resolved_path")"
	# Extract dataset name from the directory containing the JSON file
	dset=$(basename "$dsetpath")
	echo "Detected JSON file: $resolved_path"
	echo "Using directory: $dsetpath"
elif [ -d "$resolved_path" ]; then
	# It's a directory - find transforms files as before
	dsetpath="$resolved_path"
	dset=$(basename "$dataset_path")
	data0=$(find "$dsetpath" -mindepth 1 -maxdepth 1 -regex '.*/transforms_[0-9][0-9]*\.json' | sort -V | head -n 1)
	echo "Detected directory: $dsetpath"
else
	echo "Error: Dataset path does not exist or is not a valid JSON file or directory: $resolved_path"
	exit 1
fi

# Validate that we have data0
if [ -z "$data0" ] || [ ! -f "$data0" ]; then
	echo "Error: Could not find data0 JSON file"
	if [ -d "$dsetpath" ]; then
		echo "Searched in directory: $dsetpath"
	fi
	exit 1
fi

echo "mode=$mode, batch_size=$batch_size, suf=$suf, numsteps=$numsteps"
echo "Dataset path: $dsetpath"
echo "Dataset name (for outputs): $dset"
grid0=$(find "$dsetpath" -mindepth 1 -maxdepth 1 -name 'object.json' | sort -V | head -n 1)
# print these paths to the terminal
echo "###########################################################"
echo "data0: $data0"
echo "grid0: $grid0"
echo "###########################################################"

outdir="$PROJECT_ROOT/outputs"

weight_nn_width_0=20
weight_nn_width_1=20

eval_batch_size=$((batch_size / 2))

fn_ex=$(find "$dsetpath" -type f -name '*.png' | head -n 1)
downscale_factor=$(python "$SCRIPT_DIR/infer_downscale_factor.py" "$fn_ex" --target_size 250)
echo "Using downscale factor: $downscale_factor"
# Check if the downscaled images exist. If not, create them
downscaled_parent="$(dirname "$fn_ex")_$downscale_factor"
if [ ! -d "$downscaled_parent" ]; then
	echo "Creating downscaled images for dset $dset at factor $downscale_factor"
	for folder in $(find "$dsetpath" -mindepth 1 -maxdepth 1 -type d -name 'images*'); do
		python "$PROJECT_ROOT/nerf_data/scripts/resize_for_eval.py" --downscale-factor $downscale_factor --folder "$folder"
	done
fi

if [ $mode = "canonical" ]; then
	echo "Training canonical volume forward"
	python "$PROJECT_ROOT/nerfstudio/nerfstudio/scripts/train.py" nerf_xray \
		--data "$data0" \
		--output_dir "$outdir" \
		--logging.local-writer.max-log-size 10 \
		--pipeline.volumetric_supervision False \
		--pipeline.volumetric_supervision_coefficient 1e-3 \
		--pipeline.datamanager.volume_grid_file "$grid0" \
		--pipeline.datamanager.train_num_rays_per_batch $batch_size \
		--pipeline.datamanager.eval_num_rays_per_batch $eval_batch_size \
		--pipeline.model.eval_num_rays_per_chunk $eval_batch_size \
		--pipeline.flat_field_penalty 0.005 \
		--pipeline.model.flat_field_trainable False \
		--max-num-iterations $(($numsteps + 1)) \
		--optimizers.fields.scheduler.lr_pre_warmup 1e-8 \
		--optimizers.fields.scheduler.lr_final 1e-4 \
		--optimizers.fields.scheduler.warmup_steps 50 \
		--optimizers.fields.scheduler.steady_steps 2000 \
		--optimizers.fields.scheduler.max_steps $numsteps \
		--optimizers.flat_field.scheduler.lr_pre_warmup 1e-8 \
		--optimizers.flat_field.scheduler.lr_final 1e-4 \
		--optimizers.flat_field.scheduler.warmup_steps 200 \
		--optimizers.flat_field.scheduler.steady_steps 2000 \
		--optimizers.flat_field.scheduler.max_steps $numsteps \
		--timestamp "canonical_$suf" \
		multi-camera-dataparser --downscale-factors.val $downscale_factor --downscale-factors.test $downscale_factor || exit 1
	
elif [ $mode = "eval" ]; then

    dname="$outdir/$dset/$suf"
	echo "Evaluating $dname"
	config_path=$dname/config.yml

	python "$PROJECT_ROOT/nerfstudio/nerfstudio/scripts/eval.py" compute-psnr \
		--load-config "$config_path" \
		--output-path "$dname/eval_metrics_${dset}_${suf}.json"
fi