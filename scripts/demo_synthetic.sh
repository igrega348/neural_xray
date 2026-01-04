#!/bin/bash
# Demo script for synthetic data generation and training
# Generates synthetic data and runs training pipeline: canonical (forward/backward) then velocity field

set -e  # Exit on error

# Get script directory and workspace root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Dataset configuration
DSET="balls"
DATA_DIR="$WORKSPACE_ROOT/data/synthetic/$DSET"
OUTPUT_DIR="$WORKSPACE_ROOT/outputs"

# Training parameters
BATCH_SIZE=2048
BATCH_SIZE_VF=256
VF_NUM_SAMPLES_PER_RAY=256
NUMSTEPS=2000
DOWNSCALE_FACTOR=2
WEIGHT_NN_WIDTH=20
EVAL_BATCH_SIZE=$((BATCH_SIZE / 2))
EVAL_BATCH_SIZE_VF=$((BATCH_SIZE_VF / 2))
BSPLINE_METHOD='matrix'

# Velocity field parameters
VFIELD_RES_6_LRPW=1e-3
VFIELD_RES_6_WUS=1000
VFIELD_RES_6_TIMEDELTA=0.1

VFIELD_RES_12_LRPW=1e-3
VFIELD_RES_12_WUS=200
VFIELD_RES_12_TIMEDELTA=0.1

echo "=========================================="
echo "Synthetic Data Training Demo"
echo "=========================================="
echo "Workspace root: $WORKSPACE_ROOT"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Check for existing data or generate synthetic data
echo "=========================================="
echo "Step 1: Checking for existing synthetic data"
echo "=========================================="

# Define required files
REQUIRED_FILES=(
    "$DATA_DIR/transforms_00.json"
    "$DATA_DIR/transforms_20.json"
    "$DATA_DIR/transforms_00_to_20.json"
    "$DATA_DIR/balls_00.yaml"
    "$DATA_DIR/balls_20.yaml"
)

# Check if all required files exist
ALL_FILES_EXIST=true
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        ALL_FILES_EXIST=false
        MISSING_FILES+=("$file")
    fi
done

if [ "$ALL_FILES_EXIST" = true ]; then
    echo "✓ All required data files already exist"
    echo "  Skipping data generation..."
    echo ""
    echo "Found files:"
    for file in "${REQUIRED_FILES[@]}"; do
        echo "  ✓ $(basename "$file")"
    done
else
    echo "⚠ Some required data files are missing"
    echo "  Generating synthetic data..."
    echo ""
    echo "Missing files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  ✗ $(basename "$file")"
    done
    echo ""
    
    # Generate synthetic data
    bash "$SCRIPT_DIR/generate_and_animate.sh"
    
    if [ $? -ne 0 ]; then
        echo "Error: Data generation failed!"
        exit 1
    fi
    
    # Verify all files were generated
    echo ""
    echo "Verifying generated files..."
    ALL_GENERATED=true
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            echo "Error: $(basename "$file") was not generated!"
            ALL_GENERATED=false
        fi
    done
    
    if [ "$ALL_GENERATED" = false ]; then
        echo "Error: Not all required files were generated!"
        exit 1
    fi
    
    echo "✓ All required files successfully generated"
fi

# Set up paths for training
DATA0="$DATA_DIR/transforms_00.json"
DATA1="$DATA_DIR/transforms_20.json"
DATAALL="$DATA_DIR/transforms_00_to_20.json"
GRID0="$DATA_DIR/balls_00.yaml"
GRID1="$DATA_DIR/balls_20.yaml"

# Step 2: Train canonical forward
echo ""
echo "=========================================="
echo "Step 2: Training canonical volume forward"
echo "=========================================="
python "$WORKSPACE_ROOT/nerfstudio/nerfstudio/scripts/train.py" nerf_xray \
    --data "$DATA0" \
    --output_dir "$OUTPUT_DIR" \
    --logging.local-writer.max-log-size 10 \
    --pipeline.volumetric_supervision False \
    --pipeline.datamanager.volume_grid_file "$GRID0" \
    --pipeline.datamanager.train_num_rays_per_batch $BATCH_SIZE \
    --pipeline.datamanager.eval_num_rays_per_batch $EVAL_BATCH_SIZE \
    --pipeline.model.eval_num_rays_per_chunk $EVAL_BATCH_SIZE \
    --pipeline.model.flat_field_trainable False \
    --max-num-iterations $((NUMSTEPS + 1)) \
    --optimizers.fields.scheduler.lr_pre_warmup 1e-8 \
    --optimizers.fields.scheduler.lr_final 1e-4 \
    --optimizers.fields.scheduler.warmup_steps 50 \
    --optimizers.fields.scheduler.steady_steps 2000 \
    --optimizers.fields.scheduler.max_steps $NUMSTEPS \
    --timestamp "canonical_F" \
    multi-camera-dataparser --downscale-factors.val $DOWNSCALE_FACTOR --downscale-factors.test $DOWNSCALE_FACTOR || exit 1

# Step 3: Train canonical backward
echo ""
echo "=========================================="
echo "Step 3: Training canonical volume backward"
echo "=========================================="
python "$WORKSPACE_ROOT/nerfstudio/nerfstudio/scripts/train.py" nerf_xray \
    --data "$DATA1" \
    --output_dir "$OUTPUT_DIR" \
    --logging.local-writer.max-log-size 10 \
    --pipeline.volumetric_supervision False \
    --pipeline.datamanager.volume_grid_file "$GRID1" \
    --pipeline.datamanager.train_num_rays_per_batch $BATCH_SIZE \
    --pipeline.datamanager.eval_num_rays_per_batch $EVAL_BATCH_SIZE \
    --pipeline.model.eval_num_rays_per_chunk $EVAL_BATCH_SIZE \
    --pipeline.model.flat_field_trainable False \
    --max-num-iterations $((NUMSTEPS + 1)) \
    --optimizers.fields.scheduler.lr_pre_warmup 1e-8 \
    --optimizers.fields.scheduler.lr_final 1e-4 \
    --optimizers.fields.scheduler.warmup_steps 50 \
    --optimizers.fields.scheduler.steady_steps 2000 \
    --optimizers.fields.scheduler.max_steps $NUMSTEPS \
    --timestamp "canonical_B" \
    multi-camera-dataparser --downscale-factors.val $DOWNSCALE_FACTOR --downscale-factors.test $DOWNSCALE_FACTOR || exit 1

# Step 4: Train velocity field resolution 6
echo ""
echo "=========================================="
echo "Step 4: Training velocity field (resolution 6)"
echo "=========================================="

N0=6
N1=6
SUF=""
SUF2=""
STEPS=$NUMSTEPS
PADSTEPS=$(printf '%09d' $STEPS)

# Create checkpoint directory if needed
mkdir -p "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}${SUF2}/nerfstudio_models"

# Combine forward and backward checkpoints
if [ ! -f "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}${SUF2}/nerfstudio_models/step-$PADSTEPS.ckpt" ]; then
    echo "Combining forward and backward checkpoints..."
    python "$WORKSPACE_ROOT/nerfstudio-xray/nerf-xray/nerf_xray/combine_forward_backward_checkpoints.py" \
        --fwd_ckpt "$OUTPUT_DIR/$DSET/nerf_xray/canonical_F${SUF}/nerfstudio_models/step-$PADSTEPS.ckpt" \
        --bwd_ckpt "$OUTPUT_DIR/$DSET/nerf_xray/canonical_B${SUF}/nerfstudio_models/step-$PADSTEPS.ckpt" \
        --out_fn "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}${SUF2}/nerfstudio_models/step-$PADSTEPS.ckpt" || exit 1
    LOAD_OPTIMIZER=False
else
    LOAD_OPTIMIZER=True
fi

# Train velocity field
python "$WORKSPACE_ROOT/nerfstudio/nerfstudio/scripts/train.py" xray_vfield \
    --data "$DATAALL" \
    --output_dir "$OUTPUT_DIR" \
    --max-num-iterations $NUMSTEPS \
    --steps_per_eval_image 500 \
    --steps_per_save 250 \
    --logging.local-writer.max-log-size 10 \
    --load-checkpoint "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}${SUF2}/nerfstudio_models/step-$PADSTEPS.ckpt" \
    --load-optimizer $LOAD_OPTIMIZER \
    --pipeline.volumetric_supervision True \
    --pipeline.volumetric_supervision_coefficient 1e-4 \
    --pipeline.volumetric_supervision_start_step $(($NUMSTEPS+1000)) \
    --pipeline.datamanager.init_volume_grid_file "$GRID0" \
    --pipeline.datamanager.final_volume_grid_file "$GRID1" \
    --pipeline.model.deformation_field.num_control_points $N1 $N1 $N1 \
    --pipeline.model.deformation_field.weight_nn_width $WEIGHT_NN_WIDTH \
    --pipeline.model.deformation_field.weight_nn_bias True \
    --pipeline.model.deformation_field.weight_nn_gain 1.0 \
    --pipeline.model.deformation_field.timedelta $VFIELD_RES_6_TIMEDELTA \
    --pipeline.model.deformation_field.displacement_method $BSPLINE_METHOD \
    --pipeline.model.flat_field_trainable False \
    --pipeline.model.train_field_weighing False \
    --pipeline.datamanager.train_num_rays_per_batch $BATCH_SIZE_VF \
    --pipeline.datamanager.eval_num_rays_per_batch $EVAL_BATCH_SIZE_VF \
    --pipeline.model.eval_num_rays_per_chunk $EVAL_BATCH_SIZE_VF \
    --pipeline.model.num_nerf_samples_per_ray $VF_NUM_SAMPLES_PER_RAY \
    --pipeline.model.distortion_loss_mult 0.0 \
    --pipeline.model.interlevel_loss_mult 0.0 \
    --pipeline.model.disable_mixing True \
    --pipeline.density_mismatch_start_step -1 \
    --pipeline.density_mismatch_coefficient 1e-3 \
    --optimizers.fields.optimizer.lr 1e-4 \
    --optimizers.fields.optimizer.weight_decay 1e-1 \
    --optimizers.fields.scheduler.lr_pre_warmup $VFIELD_RES_6_LRPW \
    --optimizers.fields.scheduler.lr_final 1e-6 \
    --optimizers.fields.scheduler.warmup_steps $VFIELD_RES_6_WUS \
    --optimizers.fields.scheduler.steady_steps $(($NUMSTEPS - 1000)) \
    --optimizers.fields.scheduler.max_steps $NUMSTEPS \
    --timestamp "vel_${N1}${SUF2}" \
    --machine.seed 40 \
    multi-camera-dataparser --downscale-factors.val $DOWNSCALE_FACTOR --downscale-factors.test $DOWNSCALE_FACTOR || exit 1

# Step 5: Train velocity field resolution 12
echo ""
echo "=========================================="
echo "Step 5: Training velocity field (resolution 12)"
echo "=========================================="

N0=6
N1=12
STEPS=$((NUMSTEPS+NUMSTEPS))
PADSTEPS=$(printf '%09d' $STEPS)

# Create checkpoint directory if needed
mkdir -p "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}${SUF2}/nerfstudio_models"

# Refine from resolution 6 to 12 if checkpoint doesn't exist
if [ ! -f "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}${SUF2}/nerfstudio_models/step-$PADSTEPS.ckpt" ]; then
    echo "Refining velocity field from resolution $N0 to $N1..."
    python "$WORKSPACE_ROOT/nerfstudio-xray/nerf-xray/nerf_xray/refine_vfield.py" \
        --load-config "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N0}${SUF}/config.yml" \
        --new-resolution $N1 \
        --new-nn-width $WEIGHT_NN_WIDTH \
        --out-path "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}${SUF2}/nerfstudio_models/step-$PADSTEPS.ckpt" || exit 1
    LOAD_OPTIMIZER=False
else
    LOAD_OPTIMIZER=False
fi

# Train velocity field at resolution 12
python "$WORKSPACE_ROOT/nerfstudio/nerfstudio/scripts/train.py" xray_vfield \
    --data "$DATAALL" \
    --output_dir "$OUTPUT_DIR" \
    --max-num-iterations $NUMSTEPS \
    --steps_per_eval_image 500 \
    --steps_per_save 250 \
    --logging.local-writer.max-log-size 10 \
    --load-checkpoint "$OUTPUT_DIR/$DSET/xray_vfield/vel_${N1}${SUF2}/nerfstudio_models/step-$PADSTEPS.ckpt" \
    --load-optimizer $LOAD_OPTIMIZER \
    --pipeline.volumetric_supervision True \
    --pipeline.volumetric_supervision_coefficient 1e-4 \
    --pipeline.volumetric_supervision_start_step $(($NUMSTEPS+1000)) \
    --pipeline.datamanager.init_volume_grid_file "$GRID0" \
    --pipeline.datamanager.final_volume_grid_file "$GRID1" \
    --pipeline.model.deformation_field.num_control_points $N1 $N1 $N1 \
    --pipeline.model.deformation_field.weight_nn_width $WEIGHT_NN_WIDTH \
    --pipeline.model.deformation_field.weight_nn_bias True \
    --pipeline.model.deformation_field.weight_nn_gain 1.0 \
    --pipeline.model.deformation_field.timedelta $VFIELD_RES_12_TIMEDELTA \
    --pipeline.model.deformation_field.displacement_method $BSPLINE_METHOD \
    --pipeline.model.flat_field_trainable False \
    --pipeline.model.train_field_weighing False \
    --pipeline.datamanager.train_num_rays_per_batch $BATCH_SIZE_VF \
    --pipeline.datamanager.eval_num_rays_per_batch $EVAL_BATCH_SIZE_VF \
    --pipeline.model.eval_num_rays_per_chunk $EVAL_BATCH_SIZE_VF \
    --pipeline.model.num_nerf_samples_per_ray $VF_NUM_SAMPLES_PER_RAY \
    --pipeline.model.distortion_loss_mult 0.0 \
    --pipeline.model.interlevel_loss_mult 0.0 \
    --pipeline.model.disable_mixing True \
    --pipeline.density_mismatch_start_step -1 \
    --pipeline.density_mismatch_coefficient 1e-3 \
    --optimizers.fields.optimizer.lr 1e-4 \
    --optimizers.fields.optimizer.weight_decay 1e-1 \
    --optimizers.fields.scheduler.lr_pre_warmup $VFIELD_RES_12_LRPW \
    --optimizers.fields.scheduler.lr_final 1e-6 \
    --optimizers.fields.scheduler.warmup_steps $VFIELD_RES_12_WUS \
    --optimizers.fields.scheduler.steady_steps $(($NUMSTEPS - 1000)) \
    --optimizers.fields.scheduler.max_steps $NUMSTEPS \
    --timestamp "vel_${N1}${SUF2}" \
    --machine.seed 40 \
    multi-camera-dataparser --downscale-factors.val $DOWNSCALE_FACTOR --downscale-factors.test $DOWNSCALE_FACTOR || exit 1

echo ""
echo "=========================================="
echo "✓ Training complete!"
echo "=========================================="
echo "Canonical models:"
echo "  - Forward: $OUTPUT_DIR/$DSET/nerf_xray/canonical_F"
echo "  - Backward: $OUTPUT_DIR/$DSET/nerf_xray/canonical_B"
echo "Velocity field models:"
echo "  - Resolution 6: $OUTPUT_DIR/$DSET/xray_vfield/vel_6"
echo "  - Resolution 12: $OUTPUT_DIR/$DSET/xray_vfield/vel_12"
echo ""
