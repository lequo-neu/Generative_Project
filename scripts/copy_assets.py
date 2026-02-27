#!/usr/bin/env python3
"""Copy key figures to assets/ for README. Run once then delete."""
import shutil, os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets = os.path.join(root, "assets")
figures = os.path.join(root, "outputs", "figures")
samples = os.path.join(root, "outputs", "samples")

copies = [
    (os.path.join(figures, "m2_training_curves.png"), os.path.join(assets, "training_curves.png")),
    (os.path.join(figures, "m3_three_model_comparison.png"), os.path.join(assets, "three_model_comparison.png")),
    (os.path.join(figures, "m3_radar_frozen.png"), os.path.join(assets, "radar_chart.png")),
    (os.path.join(figures, "m4_failure_analysis.png"), os.path.join(assets, "failure_analysis.png")),
    (os.path.join(samples, "milestone1_sample_runs.png"), os.path.join(assets, "sample_runs.png")),
]

for src, dst in copies:
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied: {os.path.basename(src)} -> assets/")
    else:
        print(f"SKIP: {src} not found")

os.remove(__file__)
