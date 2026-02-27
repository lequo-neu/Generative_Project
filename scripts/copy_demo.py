#!/usr/bin/env python3
"""Copy demo HTML from Downloads to project. Run once."""
import shutil, os, glob

dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "index.html")
os.makedirs(os.path.dirname(dst), exist_ok=True)

candidates = glob.glob(os.path.expanduser("~/Downloads/index*.html"))
if candidates:
    src = max(candidates, key=os.path.getmtime)
    shutil.copy2(src, dst)
    print(f"Copied: {src} -> {dst}")
else:
    print(f"Download index.html from Claude, then re-run this script.")
    print(f"Target: {dst}")

os.remove(__file__)
