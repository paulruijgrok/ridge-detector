# ridge-detector-fast

A performance-optimized, drop-in replacement for `RidgeDetector` from
[`ridge-detector`](https://github.com/lxfhfut/ridge-detector) (Steger 1998
multi-scale curvilinear structure detection).

`OptimizedRidgeDetector` produces numerically identical results to the
original (verified: identical contour and point counts on test stacks) while
running roughly **4x faster** end to end:

| Stage               | Speedup |
|---------------------|--------:|
| apply_filtering      | ~5.2x  |
| compute_contours     | ~4x    |
| compute_line_width   | ~4x    |
| **total**            | **~4x**|

## Optimizations

- Analytical 2x2 eigendecomposition (`eigh_2x2`) instead of `np.linalg.eigh`
- `float32` arrays throughout (halves memory bandwidth)
- `cv2.sepFilter2D` / `cv2.filter2D` instead of `scipy.ndimage.convolve` for
  Gaussian derivative and response-surface convolutions
- Numba JIT-compiled inner contour-tracing loop

## Installation

The base `ridge-detector` package lives in `ridge-detector/` (git submodule).
Install it first, then this package in editable mode:

```bash
pip install -e ridge-detector/
pip install -e .
```

For the benchmark/evaluation scripts (matplotlib, tifffile, imageio):

```bash
pip install -e ".[scripts]"
```

## Usage

```python
from ridge_detector_fast import OptimizedRidgeDetector

det = OptimizedRidgeDetector(
    line_widths=[3],
    low_contrast=50,
    high_contrast=150,
    min_len=10,
    dark_line=False,
    estimate_width=True,
)
det.detect_lines(image)
```

`OptimizedRidgeDetector` is a subclass of `RidgeDetector` and supports the
same API (`detect_lines`, `save_results`, `show_results`, etc.).

## Project layout

```
src/ridge_detector_fast/   # the package
scripts/                    # benchmarks & evaluation tools
  test_ridge.py             # basic usage example
  detect_stack.py           # process a TIFF stack with the original detector
  detect_stack_parallel.py  # multiprocessing-based parallel stack processing
  profile_stack.py          # per-stage timing + cProfile
  compare_eigh.py           # speed/accuracy/visual comparison: original vs optimized
  test_numba_contours.py    # smoke test for compute_contours correctness/speed
ridge-detector/             # base package (git submodule)
```
