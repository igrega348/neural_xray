#!/bin/bash
# Generate synthetic X-ray data and create an animated GIF showing deformation

set -e  # Exit on error

# Get the script directory and workspace root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$WORKSPACE_ROOT/data/synthetic/balls"
OUTPUT_GIF="$DATA_DIR/deformation_animation.gif"
ROTATION_GIF="$DATA_DIR/rotation_animation.gif"

echo "=========================================="
echo "Step 1: Generating synthetic data"
echo "=========================================="

# Run the data generation script
python3 "$SCRIPT_DIR/generate_data.py"

if [ $? -ne 0 ]; then
    echo "Error: Data generation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Creating rotation animation GIF (CT)"
echo "=========================================="

IMAGES_00_DIR="$DATA_DIR/images_00"

# Check if ImageMagick is available
if command -v convert &> /dev/null; then
    echo "Using ImageMagick to create rotation GIF..."
    
    # Create list of train images in order (train_00.png through train_15.png)
    TRAIN_IMAGES=()
    for i in {0..15}; do
        IMG_PATH="$IMAGES_00_DIR/train_$(printf "%02d" $i).png"
        if [ -f "$IMG_PATH" ]; then
            TRAIN_IMAGES+=("$IMG_PATH")
        else
            echo "Warning: $IMG_PATH not found, skipping..."
        fi
    done
    
    if [ ${#TRAIN_IMAGES[@]} -eq 0 ]; then
        echo "Error: No train images found in $IMAGES_00_DIR!"
        exit 1
    fi
    
    # Create GIF with 0.1 second delay between frames (10 fps)
    convert -delay 10 -loop 0 "${TRAIN_IMAGES[@]}" "$ROTATION_GIF"
    
    echo "✓ Rotation GIF created: $ROTATION_GIF"
    echo "  Shows radiographs as the object rotates (${#TRAIN_IMAGES[@]} camera angles)"
    
elif python3 -c "import PIL" 2>/dev/null; then
    echo "Using Python PIL/Pillow to create rotation GIF..."
    
    python3 << EOF
import sys
from pathlib import Path
from PIL import Image

images_00_dir = Path("$IMAGES_00_DIR")
rotation_gif = Path("$ROTATION_GIF")

# Collect all train images in order (train_00.png through train_15.png)
train_images = []
for i in range(16):
    img_path = images_00_dir / f"train_{i:02d}.png"
    if img_path.exists():
        train_images.append(str(img_path))
    else:
        print(f"Warning: {img_path} not found, skipping...")

if not train_images:
    print(f"Error: No train images found in {images_00_dir}!")
    sys.exit(1)

# Load all images
frames = []
for img_path in train_images:
    img = Image.open(img_path)
    frames.append(img.copy())

# Save as animated GIF (100ms delay = 10 fps)
if frames:
    frames[0].save(
        str(rotation_gif),
        save_all=True,
        append_images=frames[1:],
        duration=100,  # milliseconds per frame
        loop=0  # infinite loop
    )
    print(f"✓ Rotation GIF created: {rotation_gif}")
    print(f"  Shows radiographs as the object rotates ({len(frames)} camera angles)")
else:
    print("Error: No frames to save!")
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create rotation GIF with Python!"
        exit 1
    fi
    
else
    echo "Error: Neither ImageMagick nor PIL/Pillow found!"
    echo ""
    echo "Please install one of the following:"
    echo "  - ImageMagick: brew install imagemagick (on macOS)"
    echo "  - Pillow: pip install Pillow"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 3: Creating deformation animation GIF"
echo "=========================================="

# Check if ImageMagick is available
if command -v convert &> /dev/null; then
    echo "Using ImageMagick to create GIF..."
    
    # Create list of images in order
    IMAGES=()
    for i in {0..20}; do
        IMG_PATH="$DATA_DIR/images_$(printf "%02d" $i)/eval_00.png"
        if [ -f "$IMG_PATH" ]; then
            IMAGES+=("$IMG_PATH")
        else
            echo "Warning: $IMG_PATH not found, skipping..."
        fi
    done
    
    if [ ${#IMAGES[@]} -eq 0 ]; then
        echo "Error: No eval images found!"
        exit 1
    fi
    
    # Create GIF with 0.1 second delay between frames (10 fps)
    convert -delay 10 -loop 0 "${IMAGES[@]}" "$OUTPUT_GIF"
    
    echo "✓ GIF created: $OUTPUT_GIF"
    
elif python3 -c "import PIL" 2>/dev/null; then
    echo "Using Python PIL/Pillow to create GIF..."
    
    python3 << EOF
import sys
from pathlib import Path
from PIL import Image

data_dir = Path("$DATA_DIR")
output_gif = Path("$OUTPUT_GIF")

# Collect all eval images in order
images = []
for i in range(21):
    img_path = data_dir / f"images_{i:02d}" / "eval_00.png"
    if img_path.exists():
        images.append(str(img_path))
    else:
        print(f"Warning: {img_path} not found, skipping...")

if not images:
    print("Error: No eval images found!")
    sys.exit(1)

# Load all images
frames = []
for img_path in images:
    img = Image.open(img_path)
    frames.append(img.copy())

# Save as animated GIF (100ms delay = 10 fps)
if frames:
    frames[0].save(
        str(output_gif),
        save_all=True,
        append_images=frames[1:],
        duration=100,  # milliseconds per frame
        loop=0  # infinite loop
    )
    print(f"✓ GIF created: {output_gif}")
else:
    print("Error: No frames to save!")
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create GIF with Python!"
        exit 1
    fi
    
else
    echo "Error: Neither ImageMagick nor PIL/Pillow found!"
    echo ""
    echo "Please install one of the following:"
    echo "  - ImageMagick: brew install imagemagick (on macOS)"
    echo "  - Pillow: pip install Pillow"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 4: Downsampling eval images"
echo "=========================================="

RESIZE_SCRIPT="$WORKSPACE_ROOT/nerf_data/scripts/resize_for_eval.py"

if [ ! -f "$RESIZE_SCRIPT" ]; then
    echo "Error: Resize script not found at $RESIZE_SCRIPT"
    exit 1
fi

# Process all images_XX folders
for i in {0..20}; do
    IMAGE_DIR="$DATA_DIR/images_$(printf "%02d" $i)"
    if [ -d "$IMAGE_DIR" ]; then
        echo "Downsampling images in $IMAGE_DIR..."
        python3 "$RESIZE_SCRIPT" --folder "$IMAGE_DIR" --downscale-factor 2
        if [ $? -eq 0 ]; then
            echo "  ✓ Created downsampled images in ${IMAGE_DIR}_2"
        else
            echo "  ✗ Error downsampling $IMAGE_DIR"
        fi
    else
        echo "Warning: $IMAGE_DIR not found, skipping..."
    fi
done

echo ""
echo "=========================================="
echo "✓ Complete!"
echo "=========================================="
echo "Animations created:"
echo "  - Rotation GIF (CT): $ROTATION_GIF"
echo "  - Deformation GIF: $OUTPUT_GIF"
echo "Downsampled images created in images_XX_2 folders"
echo ""

