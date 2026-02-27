import re
import random
import unicodedata
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def normalize_caption(text: str) -> str:
    """Apply basic text normalization to a caption string."""
    text = unicodedata.normalize("NFKD", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?'\-]", "", text)
    return text


def flatten_and_clean(
    dataset, subset_size: int, min_len: int, max_len: int
) -> List[Dict]:
    """Flatten multi-caption dataset and apply filtering.

    Returns a list of dicts: {image, caption, captions_all, image_id}.
    One caption is randomly selected per image; all references are kept
    in captions_all for metric computation in M3.
    """
    image_caption_map = {}

    for idx, sample in enumerate(dataset):
        image = sample["image"]
        captions = sample.get("caption", [])
        if isinstance(captions, str):
            captions = [captions]

        cleaned = []
        for cap in captions:
            c = normalize_caption(cap)
            word_count = len(c.split())
            if min_len <= word_count <= max_len:
                cleaned.append(c)

        if not cleaned:
            continue

        image_caption_map[idx] = {
            "image": image,
            "captions_all": cleaned,
            "caption": random.choice(cleaned),
            "image_id": idx,
        }

        if len(image_caption_map) >= subset_size:
            break

    records = list(image_caption_map.values())
    logger.info(f"Cleaned dataset: {len(records)} unique images")
    return records


def split_dataset(
    data: List[Dict], val_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train/val/test sets by shuffling indices."""
    n = len(data)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    train = [data[i] for i in train_idx]
    val = [data[i] for i in val_idx]
    test = [data[i] for i in test_idx]

    return train, val, test
