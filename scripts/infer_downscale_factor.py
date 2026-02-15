import argparse
from pathlib import Path
import cv2 as cv

def infer_downscale_factor(image_path: str, target_size: int) -> int:
    assert Path(image_path).is_file(), f"Image path {image_path} does not exist."
    image = cv.imread(image_path)
    height, width = image.shape[:2]
    downscale_factor = max(width, height) / target_size
    downscale_factor = int(round(downscale_factor))
    return downscale_factor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer downscale factor for an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--target_size", type=int, default=250, help="Target size for the largest dimension.")
    args = parser.parse_args()

    factor = infer_downscale_factor(args.image_path, args.target_size)
    print(factor)