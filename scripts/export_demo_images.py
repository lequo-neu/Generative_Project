#!/usr/bin/env python3
"""Export sample images from Flickr30k for the demo HTML.
Run inside the project venv after M1 notebook has been executed.

Usage: python scripts/export_demo_images.py
"""
import os
import json
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

GALLERY_PATH = os.path.join(PROJECT_ROOT, "outputs", "gallery", "all_test_captions.json")
EMBED_DIR = os.path.join(PROJECT_ROOT, "data", "embeddings")
IMG_OUT = os.path.join(PROJECT_ROOT, "demo", "images")
os.makedirs(IMG_OUT, exist_ok=True)

import torch
from datasets import load_dataset

print("Loading Flickr30k...")
raw = load_dataset("nlphuji/flickr30k", split="test", revision="refs/convert/parquet")

print("Loading test data...")
test_data = torch.load(os.path.join(EMBED_DIR, "test_embeddings.pt"), map_location="cpu", weights_only=False)

with open(GALLERY_PATH) as f:
    gallery = json.load(f)

selected = gallery
idxs = [s["idx"] for s in selected]

print(f"Exporting {len(idxs)} images...")
for s in selected:
    idx = s["idx"]
    img_id = test_data["image_ids"][idx]
    try:
        img = raw[img_id]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((400, 300))
        out_path = os.path.join(IMG_OUT, f"{idx}.jpg")
        img.save(out_path, "JPEG", quality=80)
        print(f"  Saved: {idx}.jpg")
    except Exception as e:
        print(f"  SKIP {idx}: {e}")

print(f"\nDone. {len(os.listdir(IMG_OUT))} images in {IMG_OUT}")
