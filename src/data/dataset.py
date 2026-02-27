import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


class ImageCaptionDataset(Dataset):
    """Dataset that pairs precomputed CLIP embeddings with tokenized captions.

    If embeddings are not precomputed, computes them on first access.
    For training (M2+), always precompute and cache embeddings to disk.
    """

    def __init__(
        self,
        data: List[Dict],
        clip_encoder,
        caption_tokenizer,
        precompute_embeddings: bool = True,
    ):
        self.data = data
        self.clip_encoder = clip_encoder
        self.caption_tokenizer = caption_tokenizer
        self.embeddings: Optional[torch.Tensor] = None

        if precompute_embeddings:
            self._precompute()

    def _precompute(self) -> None:
        """Precompute all CLIP embeddings to avoid redundant forward passes."""
        logger.info(f"Precomputing embeddings for {len(self.data)} images...")
        images = []
        for d in self.data:
            img = d["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        self.embeddings = self.clip_encoder.encode_batch(images, batch_size=64)
        logger.info(f"Embeddings shape: {self.embeddings.shape}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.data[idx]

        if self.embeddings is not None:
            embedding = self.embeddings[idx]
        else:
            img = record["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            embedding = self.clip_encoder.encode_image(img)

        tokens = self.caption_tokenizer.encode(record["caption"])

        return {
            "image_embedding": embedding,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "caption": record["caption"],
        }
