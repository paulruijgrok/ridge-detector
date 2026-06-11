from ridge_detector import RidgeDetector

det = RidgeDetector(
    line_widths=[3],
    low_contrast=50,
    high_contrast=150,
    min_len=10,
    dark_line=False,
    estimate_width=True,
)

det.detect_lines("/Users/paulruijgrok/Documents/Claude/Projects/ridge detection/examples/unloaded_motility/micromanager_tifs/032714/slide_2/beta_0.04mg_ml/_1/img_000000003__000.tif")
det.show_results()
