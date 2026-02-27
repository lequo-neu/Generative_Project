import logging
from typing import Dict, List

import torch
from transformers import GPT2Tokenizer

logger = logging.getLogger(__name__)


class CaptionTokenizer:
    """Wrapper around GPT-2 tokenizer for caption encoding/decoding.

    Adds a PAD token (GPT-2 does not have one by default) and configures
    padding for compatibility with batch processing.
    """

    def __init__(self, model_name: str, max_length: int):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

        logger.info(f"GPT-2 tokenizer: vocab_size={self.tokenizer.vocab_size}")

    def encode(self, caption: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single caption with padding and truncation."""
        encoded = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def encode_batch(self, captions: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of captions."""
        encoded = self.tokenizer(
            captions,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text, stripping special tokens."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
