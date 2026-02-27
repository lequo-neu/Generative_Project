import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Maps CLIP image embeddings to GPT-2 input embedding space.

    Architecture: Linear -> GELU -> Linear -> LayerNorm
    Input:  (batch, clip_dim)   e.g. (B, 512)
    Output: (batch, prefix_length, gpt2_dim) e.g. (B, 10, 768)

    The output serves as a learned prefix sequence prepended to
    GPT-2 input embeddings during caption generation.
    """

    def __init__(
        self,
        clip_dim: int = 512,
        gpt2_dim: int = 768,
        prefix_length: int = 10,
    ):
        super().__init__()
        self.prefix_length = prefix_length
        self.gpt2_dim = gpt2_dim
        hidden_dim = gpt2_dim * prefix_length

        self.projection = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, clip_embedding: torch.Tensor) -> torch.Tensor:
        projected = self.projection(clip_embedding)
        return projected.view(-1, self.prefix_length, self.gpt2_dim)
