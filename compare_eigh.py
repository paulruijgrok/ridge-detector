"""
Side-by-side comparison of original vs optimized RidgeDetector.
Evaluates:
  1. Speed       — per-stage and total timing over N frames
  2. Accuracy    — numerical difference in eigenvalues, eigenvectors, ridge positions
  3. Visual      — overlay of detected ridges saved as PNG
"""

import time
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

from ridge_detector import RidgeDetector
from optimized_detector import OptimizedRidgeDetector, eigh_2x2

STACK_PATH = "/Users/paulruijgrok/Documents/Claude/Projects/ridge detection/examples/unloaded_motility/stacks/032714/slide_2/alpha_0.04mg_ml/_1.tif"
OUT_DIR    = "/Users/paulruijgrok/Documents/Claude/Projects/ridge detection/results/comparison"
N_FRAMES   = 10  # frames used for timing

import os
os.makedirs(OUT_DIR, exist_ok=True)

PARAMS = dict(line_widths=[3], low_contrast=50, high_contrast=150,
              min_len=10, dark_line=False, estimate_width=True)

# ── Load stack ────────────────────────────────────────────────────────────────
stack = iio.imread(STACK_PATH)
print(f"Stack: {stack.shape}, dtype={stack.dtype}")

# ── 1. SPEED ──────────────────────────────────────────────────────────────────
print(f"\n=== Timing over {N_FRAMES} frames ===")

def time_detector(det, frames):
    times = {"apply_filtering": [], "compute_line_points": [],
             "compute_contours": [], "compute_line_width": [],
             "prune_contours": [], "total": []}
    for frame in frames:
        img = frame.astype(float)
        img = ((img - img.min()) / (img.max() - img.min() + 1e-12) * 255).astype(np.uint8)
        det.image = img
        det.gray  = img

        t0 = time.perf_counter()
        t = time.perf_counter(); det.apply_filtering();     times["apply_filtering"].append(time.perf_counter()-t)
        t = time.perf_counter(); det.compute_line_points(); times["compute_line_points"].append(time.perf_counter()-t)
        t = time.perf_counter(); det.compute_contours();    times["compute_contours"].append(time.perf_counter()-t)
        t = time.perf_counter(); det.compute_line_width();  times["compute_line_width"].append(time.perf_counter()-t)
        t = time.perf_counter(); det.prune_contours();      times["prune_contours"].append(time.perf_counter()-t)
        times["total"].append(time.perf_counter()-t0)
    return {k: np.mean(v) for k, v in times.items()}

frames = [stack[i] for i in range(N_FRAMES)]

orig_det = RidgeDetector(**PARAMS)
opt_det  = OptimizedRidgeDetector(**PARAMS)

t_orig = time_detector(orig_det, frames)
t_opt  = time_detector(opt_det,  frames)

print(f"{'Stage':<25} {'Original':>10} {'Optimized':>10} {'Speedup':>10}")
print("-" * 57)
for stage in ["apply_filtering", "compute_line_points", "compute_contours",
              "compute_line_width", "prune_contours", "total"]:
    o, p = t_orig[stage], t_opt[stage]
    speedup = o / p if p > 0 else float('inf')
    print(f"  {stage:<23} {o:>9.3f}s {p:>9.3f}s {speedup:>9.2f}x")

total_speedup = t_orig["total"] / t_opt["total"]
est_orig = t_orig["total"] * len(stack)
est_opt  = t_opt["total"]  * len(stack)
print(f"\nFull stack ({len(stack)} frames): {est_orig:.1f}s → {est_opt:.1f}s  ({total_speedup:.2f}x speedup)")

# ── 2. NUMERICAL ACCURACY ─────────────────────────────────────────────────────
print("\n=== Numerical accuracy (single frame) ===")

frame = stack[0]

orig_det2 = RidgeDetector(**PARAMS)
opt_det2  = OptimizedRidgeDetector(**PARAMS)

orig_det2.detect_lines(frame)
opt_det2.detect_lines(frame)

# Compare eigenvalues (saliency map)
eig_diff = np.abs(orig_det2.eigvals - opt_det2.eigvals)
print(f"  Eigenvalue map  — max diff: {eig_diff.max():.2e},  mean diff: {eig_diff.mean():.2e}")

# Compare eigenvectors (orientation map)
# Note: eigenvectors are defined up to sign, so compare |cos(angle)|
cos_sim = np.abs(np.sum(orig_det2.eigvecs * opt_det2.eigvecs, axis=2))  # dot product per pixel
vec_diff = 1 - cos_sim
print(f"  Eigenvector map — max angular diff: {vec_diff.max():.2e},  mean: {vec_diff.mean():.2e}")

# Compare number of detected ridges
n_orig = len(orig_det2.contours)
n_opt  = len(opt_det2.contours)
print(f"  Detected ridges — original: {n_orig},  optimized: {n_opt}")

# Compare ridge positions
orig_pts = np.vstack([np.stack([c.col, c.row], axis=1) for c in orig_det2.contours]) if orig_det2.contours else np.zeros((0,2))
opt_pts  = np.vstack([np.stack([c.col, c.row], axis=1) for c in opt_det2.contours])  if opt_det2.contours  else np.zeros((0,2))
print(f"  Total ridge points — original: {len(orig_pts)},  optimized: {len(opt_pts)}")

# ── 3. VISUAL COMPARISON ─────────────────────────────────────────────────────
print("\n=== Saving visual comparison ===")

def draw_ridges(image, contours, color):
    img = image.copy() if image.ndim == 3 else np.stack([image]*3, axis=-1)
    pts = [np.array([[round(c.col[j]), round(c.row[j])] for j in range(c.num)]) for c in contours]
    return cv2.polylines(img, pts, False, color, 1)

# Normalize frame for display
display = frame.astype(float)
display = ((display - display.min()) / (display.max() - display.min()) * 255).astype(np.uint8)
display_rgb = np.stack([display]*3, axis=-1)

img_orig = draw_ridges(display_rgb, orig_det2.contours, color=(255, 80,  80))   # red
img_opt  = draw_ridges(display_rgb, opt_det2.contours,  color=(80,  200, 80))   # green

# Overlay both on same image: original=red, optimized=green, overlap=yellow
overlay = display_rgb.copy()
overlay = draw_ridges(overlay, orig_det2.contours, color=(255, 60,  60))
overlay = draw_ridges(overlay, opt_det2.contours,  color=(60,  255, 60))

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
axes[0].imshow(img_orig);   axes[0].set_title("Original (red ridges)",   fontsize=13); axes[0].axis("off")
axes[1].imshow(img_opt);    axes[1].set_title("Optimized (green ridges)", fontsize=13); axes[1].axis("off")
axes[2].imshow(overlay);    axes[2].set_title("Overlay (red=orig, green=opt)", fontsize=13); axes[2].axis("off")

red_patch   = mpatches.Patch(color=(1, 0.24, 0.24), label=f"Original  ({n_orig} ridges)")
green_patch = mpatches.Patch(color=(0.24, 1, 0.24), label=f"Optimized ({n_opt} ridges)")
axes[2].legend(handles=[red_patch, green_patch], loc="lower right", fontsize=11)

plt.suptitle(f"Ridge detection comparison — frame 0\n"
             f"Original: {t_orig['total']:.3f}s/frame  |  "
             f"Optimized: {t_opt['total']:.3f}s/frame  |  "
             f"Speedup: {total_speedup:.2f}x",
             fontsize=12)
plt.tight_layout()
out_path = f"{OUT_DIR}/comparison_frame0.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved to {out_path}")
