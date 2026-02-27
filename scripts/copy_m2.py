#!/usr/bin/env python3
"""Copy M2 notebook from Downloads to project notebooks folder."""
import shutil, os, glob

dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "milestone2_model_training.ipynb")
candidates = glob.glob(os.path.expanduser("~/Downloads/*milestone2*model*training*.ipynb"))

if candidates:
    src = max(candidates, key=os.path.getmtime)
    shutil.copy2(src, dst)
    print(f"Copied: {src} -> {dst}")
else:
    print("Not found in Downloads. Please manually copy milestone2_model_training.ipynb to:")
    print(f"  {dst}")

os.remove(__file__)
