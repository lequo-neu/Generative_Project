import logging
from typing import List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class CLIPImageEncoder:
    """Wrapper around CLIP ViT-B/32 for image embedding extraction.

    The encoder runs in eval mode with gradients disabled at all times.
    Embeddings are L2-normalized to unit length following CLIP convention.
    """

    def __init__(self, model_name: str, device: str):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"CLIP loaded: {model_name} on {device} ({total_params:,} params)")

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a single PIL image into a 512-d embedding vector."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        embedding = self.model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze(0).cpu()

    @torch.no_grad()
    def encode_batch(
        self, images: List[Image.Image], batch_size: int = 32
    ) -> torch.Tensor:
        """Encode a list of PIL images into a (N, 512) embedding matrix."""
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(
                images=batch, return_tensors="pt", padding=True
            ).to(self.device)
            embeddings = self.model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)
