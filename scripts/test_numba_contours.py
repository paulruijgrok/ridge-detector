"""
Quick smoke-test for the Numba-accelerated compute_contours.

Checks:
  1. Numba kernel compiles and runs
  2. Number of detected contours is close to original (±5%)
  3. Per-stage timing comparison over 5 frames
"""
import time
import numpy as np
import imageio.v3 as iio

STACK_PATH = "/Users/paulruijgrok/Documents/Claude/Projects/ridge detection/examples/unloaded_motility/stacks/032714/slide_2/alpha_0.04mg_ml/_1.tif"
PARAMS = dict(line_widths=[3], low_contrast=50, high_contrast=150,
              min_len=10, dark_line=False, estimate_width=True)

stack = iio.imread(STACK_PATH)
frame = stack[0]
img   = frame.astype(float)
img   = ((img - img.min()) / (img.max() - img.min() + 1e-12) * 255).astype(np.uint8)

# ── Import both detectors ──────────────────────────────────────────────────────
from ridge_detector import RidgeDetector
from ridge_detector_fast import OptimizedRidgeDetector

orig = RidgeDetector(**PARAMS)
opt  = OptimizedRidgeDetector(**PARAMS)

# ── Correctness check (single frame) ──────────────────────────────────────────
print("=== Correctness check ===")
print("Running original...")
orig.detect_lines(img)
n_orig = len(orig.contours)

print("Running optimized (first run JIT-compiles the Numba kernel)...")
t0 = time.perf_counter()
opt.detect_lines(img)
t_first = time.perf_counter() - t0
n_opt  = len(opt.contours)

print(f"  Original contours : {n_orig}")
print(f"  Optimized contours: {n_opt}")
pct_diff = abs(n_orig - n_opt) / max(n_orig, 1) * 100
status = "OK" if pct_diff < 5 else "WARNING (>5% difference)"
print(f"  Difference: {pct_diff:.1f}%  →  {status}")
print(f"  (First-run time including JIT compile: {t_first:.2f}s)")

# ── Per-stage timing (warm, 5 frames) ─────────────────────────────────────────
print("\n=== Per-stage timing (avg over 5 warm frames) ===")

def stage_times(det, frames):
    stages = ["apply_filtering", "compute_line_points", "compute_contours",
              "compute_line_width", "prune_contours", "total"]
    times  = {s: [] for s in stages}
    for f in frames:
        img_ = f.astype(float)
        img_ = ((img_ - img_.min()) / (img_.max() - img_.min() + 1e-12) * 255).astype(np.uint8)
        det.image = img_
        det.gray  = img_
        t0 = time.perf_counter()
        for s in stages[:-1]:
            t = time.perf_counter(); getattr(det, s)(); times[s].append(time.perf_counter()-t)
        times["total"].append(time.perf_counter()-t0)
    return {s: np.mean(v) for s, v in times.items()}

frames5 = [stack[i] for i in range(5)]
t_orig = stage_times(orig, frames5)
t_opt  = stage_times(opt,  frames5)

print(f"{'Stage':<25} {'Original':>10} {'Optimized':>10} {'Speedup':>10}")
print("-" * 57)
for s in ["apply_filtering", "compute_line_points", "compute_contours",
          "compute_line_width", "prune_contours", "total"]:
    o, p = t_orig[s], t_opt[s]
    x    = o / p if p > 0 else float('inf')
    print(f"  {s:<23} {o:>9.3f}s {p:>9.3f}s {x:>9.2f}x")

overall = t_orig["total"] / t_opt["total"]
print(f"\nOverall speedup: {overall:.2f}x")
est_orig = t_orig["total"] * len(stack)
est_opt  = t_opt["total"]  * len(stack)
print(f"Full stack estimate: {est_orig:.1f}s → {est_opt:.1f}s")
