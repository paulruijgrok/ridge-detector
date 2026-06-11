"""
Parallel stack processing using multiprocessing.

Each frame is processed independently on a separate CPU core.
Workers read their own frame directly from disk to avoid serialising
large arrays over IPC.

Usage:
    python detect_stack_parallel.py

The `if __name__ == '__main__':` guard is required on macOS (spawn start method).
"""

import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

STACK_PATH = ("/Users/paulruijgrok/Documents/Claude/Projects/ridge detection/"
              "examples/unloaded_motility/stacks/032714/slide_2/alpha_0.04mg_ml/_1.tif")

PARAMS = dict(line_widths=[3], low_contrast=50, high_contrast=150,
              min_len=10, dark_line=False, estimate_width=True)


# ── Worker function (must be module-level for pickling on macOS spawn) ────────

def _detect_frame(args):
    """Load one frame by index, run detection, return (frame_idx, contours)."""
    stack_path, frame_idx, params = args

    # Imports inside worker to avoid issues with spawn on macOS
    import numpy as np
    import tifffile
    from ridge_detector_fast import OptimizedRidgeDetector

    # tifffile reliably returns (H, W) for grayscale TIFFs
    with tifffile.TiffFile(stack_path) as tif:
        frame = tif.pages[frame_idx].asarray()

    img = frame.astype(float)
    img = ((img - img.min()) / (img.max() - img.min() + 1e-12) * 255).astype(np.uint8)

    # Set image/gray directly to bypass detect_lines' channel-check
    det = OptimizedRidgeDetector(**params)
    det.image = img
    det.gray  = img
    det.apply_filtering()
    det.compute_line_points()
    det.compute_contours()
    det.compute_line_width()
    det.prune_contours()
    return frame_idx, det.contours


# ── Reusable API ──────────────────────────────────────────────────────────────

def detect_stack_parallel(stack_path, params, n_workers=None, n_frames=None):
    """
    Process all frames of a multi-frame TIFF in parallel.

    Returns a list of (frame_idx, contours) sorted by frame index.
    """
    if n_workers is None:
        n_workers = os.cpu_count()
    if n_frames is None:
        import tifffile
        with tifffile.TiffFile(stack_path) as tif:
            n_frames = len(tif.pages)

    args = [(stack_path, i, params) for i in range(n_frames)]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_detect_frame, args))
    return sorted(results, key=lambda x: x[0])


# ── Benchmark script ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    n_workers = os.cpu_count()
    import tifffile
    with tifffile.TiffFile(STACK_PATH) as tif:
        n_frames = len(tif.pages)
        shape = tif.pages[0].asarray().shape
    print(f"Stack: {n_frames} frames × {shape}, {n_workers} CPU cores available")

    # 1. Warm up Numba cache in the main process so workers load from disk cache
    #    (avoids recompilation in every worker on first call)
    print("\nWarming up Numba cache (one frame)...")
    t0 = time.perf_counter()
    _detect_frame((STACK_PATH, 0, PARAMS))
    print(f"  Done in {time.perf_counter()-t0:.2f}s (includes JIT compile if first run)")

    # 2. Serial baseline — 5 frames
    print("\nSerial baseline (5 frames)...")
    t0 = time.perf_counter()
    for i in range(5):
        _detect_frame((STACK_PATH, i, PARAMS))
    t_per_frame  = (time.perf_counter() - t0) / 5
    t_serial_est = t_per_frame * n_frames
    print(f"  {t_per_frame:.3f}s / frame  →  ~{t_serial_est:.1f}s for full stack")

    # 3. Parallel — all frames
    print(f"\nParallel ({n_workers} workers, {n_frames} frames)...")
    args = [(STACK_PATH, i, PARAMS) for i in range(n_frames)]
    t0  = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_detect_frame, args))
    t_parallel = time.perf_counter() - t0

    results.sort(key=lambda x: x[0])
    n_contours = [len(r[1]) for r in results]

    speedup = t_serial_est / t_parallel
    print(f"  {t_parallel:.1f}s total  ({t_parallel/n_frames:.3f}s / frame)")
    print(f"\n{'─'*45}")
    print(f"  Serial  (estimated)  {t_serial_est:>6.1f}s")
    print(f"  Parallel             {t_parallel:>6.1f}s")
    print(f"  Speedup              {speedup:>6.2f}x  ({n_workers} cores)")
    print(f"{'─'*45}")
    print(f"  Contours/frame: min={min(n_contours)}  max={max(n_contours)}  "
          f"mean={sum(n_contours)/len(n_contours):.1f}")
