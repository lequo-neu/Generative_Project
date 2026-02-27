#!/usr/bin/env python3
"""Copy key figures to assets/ for README. Run once."""
import shutil, os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets = os.path.join(root, "assets")
os.makedirs(assets, exist_ok=True)

copies = [
    ("outputs/figures/m2_training_curves.png", "training_curves.png"),
    ("outputs/figures/m3_three_model_comparison.png", "three_model_comparison.png"),
    ("outputs/figures/m3_radar_frozen.png", "radar_chart.png"),
    ("outputs/figures/m4_failure_analysis.png", "failure_analysis.png"),
    ("outputs/samples/milestone1_sample_runs.png", "sample_runs.png"),
]

for src_rel, dst_name in copies:
    src = os.path.join(root, src_rel)
    dst = os.path.join(assets, dst_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  OK: {src_rel} -> assets/{dst_name}")
    else:
        print(f"  SKIP: {src_rel} not found")

print("\nDone. Delete this script after running.")
