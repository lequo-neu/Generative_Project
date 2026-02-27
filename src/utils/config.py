import os
from dataclasses import dataclass
from typing import Optional

import torch
import yaml


@dataclass
class Config:
    """Central configuration for the entire project pipeline."""

    project_root: str = os.path.expanduser(
        "~/Documents/GitHub/Python/VESKL/11.DAE/NEU/NEU_IE7615/Prj/Generative"
    )
    data_raw_dir: str = ""
    data_processed_dir: str = ""
    embeddings_dir: str = ""
    outputs_dir: str = ""

    dataset_name: str = "flickr30k"
    hf_dataset_id: str = "nlphuji/flickr30k"
    subset_size: int = 3000
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    min_caption_length: int = 8
    max_caption_length: int = 60

    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_embedding_dim: int = 512

    gpt2_model_name: str = "gpt2"
    gpt2_embedding_dim: int = 768
    max_token_length: int = 50

    seed: int = 42
    device: str = ""

    def __post_init__(self):
        self.data_raw_dir = os.path.join(self.project_root, "data", "raw")
        self.data_processed_dir = os.path.join(self.project_root, "data", "processed")
        self.embeddings_dir = os.path.join(self.project_root, "data", "embeddings")
        self.outputs_dir = os.path.join(self.project_root, "outputs")

        if not self.device:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from a YAML file, overriding defaults."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        kwargs = {}
        if "dataset" in raw:
            d = raw["dataset"]
            kwargs.update({
                "dataset_name": d.get("name", cls.dataset_name),
                "hf_dataset_id": d.get("hf_id", cls.hf_dataset_id),
                "subset_size": d.get("subset_size", cls.subset_size),
                "val_ratio": d.get("val_ratio", cls.val_ratio),
                "test_ratio": d.get("test_ratio", cls.test_ratio),
                "min_caption_length": d.get("min_caption_length", cls.min_caption_length),
                "max_caption_length": d.get("max_caption_length", cls.max_caption_length),
            })
        if "clip" in raw:
            kwargs["clip_model_name"] = raw["clip"].get("model_name", cls.clip_model_name)
            kwargs["clip_embedding_dim"] = raw["clip"].get("embedding_dim", cls.clip_embedding_dim)
        if "gpt2" in raw:
            kwargs["gpt2_model_name"] = raw["gpt2"].get("model_name", cls.gpt2_model_name)
            kwargs["gpt2_embedding_dim"] = raw["gpt2"].get("embedding_dim", cls.gpt2_embedding_dim)
            kwargs["max_token_length"] = raw["gpt2"].get("max_token_length", cls.max_token_length)
        if "seed" in raw:
            kwargs["seed"] = raw["seed"]

        return cls(**kwargs)
