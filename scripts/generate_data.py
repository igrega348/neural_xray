#!/usr/bin/env python3
"""
Generate synthetic X-ray projection data with 8 spheres that deform over time.
"""

import sys
import json
import yaml
from pathlib import Path
import tempfile
import shutil

# Add xray_projection_render to path
sys.path.insert(0, str(Path(__file__).parent.parent / "xray_projection_render"))
from xray_renderer import XRayRenderer


def apply_deformation(X_1, X_2, X_3, t):
    """
    Apply deformation field to original coordinates.
    
    Args:
        X_1, X_2, X_3: Original coordinates
        t: Time parameter (0 to 1)
    
    Returns:
        Deformed coordinates (x_1, x_2, x_3)
    """
    # relu(X) = max(0, X)
    relu_X1 = max(0, X_1)
    relu_X3 = max(0, X_3)
    
    x_1 = X_1 + 0.5 * relu_X1 * t
    x_2 = X_2 + relu_X3 * X_2 * t
    x_3 = X_3 + X_1 * X_2 * X_3 * t
    
    return x_1, x_2, x_3


def create_sphere_collection(sphere_centers):
    """
    Create an object_collection YAML structure from sphere centers.
    
    Args:
        sphere_centers: List of (x, y, z) tuples for sphere centers
    
    Returns:
        Dictionary representing object_collection
    """
    objects = []
    for center in sphere_centers:
        objects.append({
            "center": list(center),
            "radius": 0.15,
            "rho": 1.0,
            "type": "sphere"
        })
    
    return {
        "type": "object_collection",
        "objects": objects
    }


def main():
    # Setup and initialization
    print("Initializing renderer...")
    try:
        renderer = XRayRenderer()
        print("✓ Successfully loaded the shared library")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease build the shared library first:")
        print("  cd xray_projection_render")
        print("  ./build.sh")
        sys.exit(1)
    
    # Define output directory
    workspace_root = Path(__file__).parent.parent
    output_dir = workspace_root / "data" / "synthetic" / "balls"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image subdirectories
    for i in range(21):
        (output_dir / f"images_{i:02d}").mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Generate initial configuration (t=0)
    print("\nGenerating initial sphere configuration...")
    sphere_coords = [
        (-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5),
        (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5),
        (0.5, -0.5, -0.5), (0.5, -0.5, 0.5),
        (0.5, 0.5, -0.5), (0.5, 0.5, 0.5),
    ]
    
    initial_collection = create_sphere_collection(sphere_coords)
    config_file = output_dir / "balls.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(initial_collection, f, default_flow_style=False)
    print(f"✓ Saved initial config to {config_file}")
    
    # Also save as balls_00.yaml for volume grid
    balls_00_file = output_dir / "balls_00.yaml"
    with open(balls_00_file, 'w') as f:
        yaml.dump(initial_collection, f, default_flow_style=False)
    print(f"✓ Saved initial config to {balls_00_file}")
    
    # Store original sphere centers for deformation
    original_centers = sphere_coords.copy()
    
    # Rendering loop for all timesteps
    print("\nStarting rendering loop...")
    all_transforms = {}  # Store transforms for aggregation
    final_collection = None  # Store final deformed state for balls_20.yaml
    
    for i in range(21):
        t = i * 0.05  # t=0 for i=0, t=1.0 for i=20
        print(f"\nTimestep {i:02d} (t={t:.2f})...")
        
        # Apply deformation if t > 0
        if t > 0:
            deformed_centers = [apply_deformation(X_1, X_2, X_3, t) 
                              for X_1, X_2, X_3 in original_centers]
            collection = create_sphere_collection(deformed_centers)
            # Store final state for balls_20.yaml
            if i == 20:
                final_collection = collection
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(collection, temp_file, default_flow_style=False)
            temp_file.close()
            input_file = temp_file.name
        else:
            input_file = str(config_file)
        
        # Set up render parameters
        params = {
            'input': input_file,
            'output_dir': str(output_dir / f"images_{i:02d}"),
            'fname_pattern': 'train_%02d.png',
            'resolution': 500,
            'R': 4.0,
            'fov': 40.0,
            'time_label': t,
            'log_level': 'error',  # Quiet operation
        }
        
        # Configure camera angles based on timestep
        if i == 0 or i == 20:
            # 16 equispaced projections for t=0 and t=1
            params['num_images'] = 16
            params['polar_angle'] = 90.0
            camera_angles = None
        else:
            # 2 projections for intermediate timesteps
            # Use 90° and 180° to match the equispaced sequence (first and 5th angles)
            camera_angles = [
                {'azimuthal': 90, 'polar': 90},
                {'azimuthal': 180, 'polar': 90}
            ]
        
        # Always generate transforms file for all timesteps
        transforms_file = str(output_dir / f"transforms_{i:02d}.json")
        params['transforms_file'] = transforms_file
        
        # Render projections
        try:
            if camera_angles is None:
                result = renderer.render(params)
            else:
                result = renderer.render(params, camera_angles=camera_angles)
            
            if result['success']:
                print(f"  ✓ Rendered {result['num_images']} images")
                # Load transforms from generated file
                if Path(transforms_file).exists():
                    with open(transforms_file, 'r') as f:
                        all_transforms[i] = json.load(f)
            else:
                print(f"  ✗ Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  ✗ Exception during rendering: {e}")
        
        # Clean up temporary file if created
        if t > 0 and Path(input_file).exists():
            Path(input_file).unlink()
    
    # Create eval images (copy train_00.png to eval_00.png for each timestep)
    print("\nCreating eval images...")
    for i in range(21):
        image_dir = output_dir / f"images_{i:02d}"
        train_00 = image_dir / "train_00.png"
        eval_00 = image_dir / "eval_00.png"
        
        if train_00.exists():
            shutil.copy2(train_00, eval_00)
            print(f"  ✓ Created {eval_00}")
    
    # Assemble transform files
    print("\nAssembling transform files...")
    
    # transforms_00.json - copy from timestep 0
    if 0 in all_transforms:
        transforms_00 = all_transforms[0].copy()
        # Update file paths to include images_00/ prefix
        for frame in transforms_00['frames']:
            if not frame['file_path'].startswith('images_00/'):
                frame['file_path'] = f"images_00/{Path(frame['file_path']).name}"
        
        # Add eval frame (copy of first frame)
        if transforms_00['frames']:
            eval_frame = transforms_00['frames'][0].copy()
            eval_frame['file_path'] = 'images_00/eval_00.png'
            transforms_00['frames'].append(eval_frame)
        
        transforms_00_file = output_dir / "transforms_00.json"
        with open(transforms_00_file, 'w') as f:
            json.dump(transforms_00, f, indent=2)
        print(f"✓ Created {transforms_00_file}")
    
    # transforms_20.json - copy from timestep 20
    if 20 in all_transforms:
        transforms_20 = all_transforms[20].copy()
        # Update file paths to include images_20/ prefix
        for frame in transforms_20['frames']:
            if not frame['file_path'].startswith('images_20/'):
                frame['file_path'] = f"images_20/{Path(frame['file_path']).name}"
        
        # Add eval frame (copy of first frame)
        if transforms_20['frames']:
            eval_frame = transforms_20['frames'][0].copy()
            eval_frame['file_path'] = 'images_20/eval_00.png'
            transforms_20['frames'].append(eval_frame)
        
        transforms_20_file = output_dir / "transforms_20.json"
        with open(transforms_20_file, 'w') as f:
            json.dump(transforms_20, f, indent=2)
        print(f"✓ Created {transforms_20_file}")
    
    # transforms.json - aggregate all timesteps
    print("Aggregating transforms from all timesteps...")
    
    if 0 not in all_transforms:
        raise RuntimeError("No transforms file found for timestep 0. Cannot create aggregated transforms.")
    
    # Copy the whole JSON from transforms_00.json as the base
    aggregated_transforms = all_transforms[0].copy()
    
    # Update timestep 0 frames to ensure correct paths and time
    for frame in aggregated_transforms['frames']:
        if not frame['file_path'].startswith('images_00/'):
            frame['file_path'] = f"images_00/{Path(frame['file_path']).name}"
        frame['time'] = 0.0
    
    # Add eval frame for timestep 0
    if aggregated_transforms['frames']:
        eval_frame = aggregated_transforms['frames'][0].copy()
        eval_frame['file_path'] = 'images_00/eval_00.png'
        aggregated_transforms['frames'].append(eval_frame)
    
    # Extend frames with frames from all other timesteps
    for i in range(1, 21):
        t = i * 0.05
        if i in all_transforms:
            for frame in all_transforms[i]['frames']:
                # Ensure path is relative to output_dir
                if not frame['file_path'].startswith(f'images_{i:02d}/'):
                    frame['file_path'] = f"images_{i:02d}/{Path(frame['file_path']).name}"
                frame['time'] = t
                aggregated_transforms['frames'].append(frame)
            
            # Add eval frame for this timestep (copy of first frame)
            if all_transforms[i]['frames']:
                eval_frame = all_transforms[i]['frames'][0].copy()
                eval_frame['file_path'] = f"images_{i:02d}/eval_00.png"
                eval_frame['time'] = t
                aggregated_transforms['frames'].append(eval_frame)
    
    # Save aggregated transforms as transforms_00_to_20.json
    transforms_file = output_dir / "transforms_00_to_20.json"
    with open(transforms_file, 'w') as f:
        json.dump(aggregated_transforms, f, indent=2)
    print(f"✓ Created {transforms_file} with {len(aggregated_transforms['frames'])} frames")
    
    # Clean up intermediate transforms files (keep only 00, 20, and aggregated)
    print("\nCleaning up intermediate transforms files...")
    for i in range(21):
        if i != 0 and i != 20:
            transforms_file = output_dir / f"transforms_{i:02d}.json"
            if transforms_file.exists():
                transforms_file.unlink()
                print(f"  ✓ Removed {transforms_file}")
    
    # Save final deformed state as balls_20.yaml
    if final_collection is not None:
        balls_20_file = output_dir / "balls_20.yaml"
        with open(balls_20_file, 'w') as f:
            yaml.dump(final_collection, f, default_flow_style=False)
        print(f"✓ Saved final deformed state to {balls_20_file}")
    else:
        print("⚠ Warning: Final collection not saved. balls_20.yaml will not be created.")
    
    # Clean up object.json if it was generated
    object_json_file = output_dir / "object.json"
    if object_json_file.exists():
        object_json_file.unlink()
        print(f"✓ Removed {object_json_file}")
    
    print("\n✓ Data generation complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Config file: {config_file}")
    print(f"  Transform files: transforms_00.json, transforms_20.json, transforms_00_to_20.json")
    print(f"  Volume grid files: balls_00.yaml, balls_20.yaml")


if __name__ == '__main__':
    main()
