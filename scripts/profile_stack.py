import cProfile
import pstats
import io
import time
import imageio.v3 as iio
import numpy as np
from ridge_detector import RidgeDetector

stack_path = "/Users/paulruijgrok/Documents/Claude/Projects/ridge detection/examples/unloaded_motility/stacks/032714/slide_2/alpha_0.04mg_ml/_1.tif"

stack = iio.imread(stack_path)
frame = stack[0]  # Profile on a single frame

det = RidgeDetector(
    line_widths=[3],
    low_contrast=50,
    high_contrast=150,
    min_len=10,
    dark_line=False,
    estimate_width=True,
)

# --- Per-stage timing (5 frames to average) ---
n_frames = 5
times = {
    "apply_filtering": [],
    "compute_line_points": [],
    "compute_contours": [],
    "compute_line_width": [],
    "prune_contours": [],
    "total": [],
}

for i in range(n_frames):
    f = stack[i]

    # Replicate detect_lines preprocessing
    import cv2
    img = f.astype(float)
    img = ((img - img.min()) / (img.max() - img.min() + 1e-12) * 255).astype(np.uint8)
    det.image = img
    det.gray = img  # already grayscale

    t0 = time.perf_counter()

    t = time.perf_counter(); det.apply_filtering();      times["apply_filtering"].append(time.perf_counter() - t)
    t = time.perf_counter(); det.compute_line_points();  times["compute_line_points"].append(time.perf_counter() - t)
    t = time.perf_counter(); det.compute_contours();     times["compute_contours"].append(time.perf_counter() - t)
    t = time.perf_counter(); det.compute_line_width();   times["compute_line_width"].append(time.perf_counter() - t)
    t = time.perf_counter(); det.prune_contours();       times["prune_contours"].append(time.perf_counter() - t)

    times["total"].append(time.perf_counter() - t0)

print("=== Per-stage timing (avg over 5 frames) ===")
total_avg = np.mean(times["total"])
for stage, vals in times.items():
    avg = np.mean(vals)
    pct = 100 * avg / total_avg if stage != "total" else 100
    print(f"  {stage:<25} {avg:.3f}s  ({pct:.1f}%)")

est_total = total_avg * len(stack)
print(f"\nEstimated time for full stack ({len(stack)} frames): {est_total:.1f}s ({est_total/60:.1f} min)")

# --- cProfile on a single frame for function-level detail ---
print("\n=== cProfile: top 20 functions by cumulative time ===")
pr = cProfile.Profile()
pr.enable()
det.detect_lines(frame)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(20)
print(s.getvalue())
