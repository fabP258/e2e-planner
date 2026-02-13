"""Spatial transformer encoder for processing patches within each frame."""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Standard transformer encoder block with pre-norm architecture."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # MLP block
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, N, D] where N is sequence length

        Returns:
            Output tensor of shape [B, N, D]
        """
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class SpatialTransformerEncoder(nn.Module):
    """
    Spatial transformer encoder that processes patches within each frame.

    Applies self-attention over patches of a single frame to capture
    spatial relationships (road, lanes, objects).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Patch embeddings of shape [B, num_frames, num_patches, embed_dim]
               or [B, num_patches, embed_dim] for a single frame

        Returns:
            Encoded features of same shape as input
        """
        # Handle both batched frames and single frame inputs
        if x.dim() == 4:
            # Input: [B, num_frames, num_patches, embed_dim]
            B, T, N, D = x.shape

            # Reshape to process all frames in parallel
            # [B, T, N, D] -> [B*T, N, D]
            x = x.view(B * T, N, D)

            # Apply transformer blocks
            for block in self.blocks:
                x = block(x)

            # Apply final norm
            x = self.norm(x)

            # Reshape back: [B*T, N, D] -> [B, T, N, D]
            x = x.view(B, T, N, D)

        else:
            # Input: [B, N, D] - single frame
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)

        return x
