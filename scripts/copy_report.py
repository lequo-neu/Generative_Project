#!/usr/bin/env python3
"""Copy the generated PDF report to the reports folder.
Run: python scripts/copy_report.py
"""
import shutil, os

src = os.path.expanduser("~/Downloads/Milestone1_Report_G8-Generative.pdf")
dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "Milestone1_Report_G8-Generative.pdf")

# Try multiple possible download locations
candidates = [
    src,
    os.path.expanduser("~/Downloads/Milestone1 Report G8-Generative.pdf"),
]

for c in candidates:
    if os.path.exists(c):
        shutil.copy2(c, dst)
        print(f"Copied: {c} -> {dst}")
        break
else:
    print("PDF not found in Downloads. Please manually copy the PDF from Claude's output to:")
    print(f"  {dst}")
