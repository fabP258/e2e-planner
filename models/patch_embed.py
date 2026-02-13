"""Patch embedding with positional encoding for vision transformer."""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts images into patch embeddings with positional encoding.

    Takes an image and:
    1. Splits it into non-overlapping patches
    2. Projects each patch to embedding dimension via linear layer
    3. Adds learnable positional embeddings
    4. Optionally prepends a CLS token
    """

    def __init__(
        self,
        image_height: int = 128,
        image_width: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        use_cls_token: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # Calculate number of patches
        self.num_patches_h = image_height // patch_size
        self.num_patches_w = image_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patch embedding: Conv2d with kernel_size=stride=patch_size
        # This is equivalent to splitting into patches and applying linear projection
        self.patch_proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # CLS token (learnable)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_tokens = self.num_patches + 1
        else:
            self.cls_token = None
            num_tokens = self.num_patches

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize positional embedding with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize patch projection
        w = self.patch_proj.weight.data
        nn.init.xavier_uniform_(w.view(w.size(0), -1))
        nn.init.zeros_(self.patch_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images of shape [B, C, H, W]

        Returns:
            Patch embeddings of shape [B, num_patches (+1 if CLS), embed_dim]
        """
        B, C, H, W = x.shape

        # Project patches: [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.patch_proj(x)

        # Flatten spatial dimensions: [B, embed_dim, H/P, W/P] -> [B, embed_dim, num_patches]
        x = x.flatten(2)

        # Transpose to [B, num_patches, embed_dim]
        x = x.transpose(1, 2)

        # Prepend CLS token if used
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply dropout
        x = self.dropout(x)

        return x

    def get_num_tokens(self) -> int:
        """Returns total number of tokens (patches + optional CLS)."""
        if self.use_cls_token:
            return self.num_patches + 1
        return self.num_patches
