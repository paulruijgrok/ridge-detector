import os
import imageio.v3 as iio
import numpy as np
from ridge_detector import RidgeDetector

stack_path = "/Users/paulruijgrok/Documents/Claude/Projects/ridge detection/examples/unloaded_motility/stacks/032714/slide_2/alpha_0.04mg_ml/_1.tif"
output_dir = "/Users/paulruijgrok/Documents/Claude/Projects/ridge detection/results/alpha_stack"
os.makedirs(output_dir, exist_ok=True)

stack = iio.imread(stack_path)  # shape: (n_frames, H, W)
print(f"Stack shape: {stack.shape}, dtype: {stack.dtype}")

det = RidgeDetector(
    line_widths=[3],
    low_contrast=50,
    high_contrast=150,
    min_len=10,
    dark_line=False,
    estimate_width=True,
)

for i, frame in enumerate(stack):
    print(f"Processing frame {i+1}/{len(stack)}...", end="\r")
    det.detect_lines(frame)
    det.save_results(save_dir=output_dir, prefix=f"frame_{i:04d}")

print(f"\nDone. Results saved to {output_dir}")
