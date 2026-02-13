"""Temporal transformer encoder for reasoning across frames."""

import torch
import torch.nn as nn


class TemporalTransformerEncoder(nn.Module):
    """
    Temporal transformer encoder that attends across frames.

    Processes the sequence of frames to capture temporal dynamics
    for trajectory prediction.

    Two modes:
    - CLS token mode: Attends over CLS tokens from each frame (efficient)
    - Full mode: Attends over all patches across all frames (richer but expensive)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_frames: int = 6,
        use_cls_only: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.use_cls_only = use_cls_only

        # Learnable temporal positional embedding
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim)
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize temporal positional embedding
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Spatial features of shape [B, num_frames, num_patches, embed_dim]
               If use_cls_only=True, expects CLS token at position 0
            return_sequence: If True, return full sequence; else return pooled

        Returns:
            If return_sequence: [B, num_frames, embed_dim]
            Else: [B, embed_dim] (pooled representation)
        """
        B, T, N, D = x.shape

        if self.use_cls_only:
            # Extract CLS tokens (assumed to be at index 0)
            # [B, T, N, D] -> [B, T, D]
            x = x[:, :, 0, :]
            # Add temporal positional embedding
            x = x + self.temporal_pos_embed
        else:
            # Flatten all patches across time
            # [B, T, N, D] -> [B, T*N, D]
            x = x.view(B, T * N, D)

            # For full attention, we need to add temporal position info
            # Create temporal indices for each patch
            temporal_pos = self.temporal_pos_embed.repeat(1, N, 1)
            temporal_pos = temporal_pos.view(1, T * N, D)
            x = x + temporal_pos.expand(B, -1, -1)

        # Apply transformer encoder
        x = self.transformer(x)

        # Apply final norm
        x = self.norm(x)

        if return_sequence:
            if self.use_cls_only:
                return x  # [B, T, D]
            else:
                # Reshape back if needed
                return x.view(B, T, N, D).mean(dim=2)  # [B, T, D]
        else:
            # Pool across temporal dimension
            # Mean pooling over frames
            if self.use_cls_only:
                return x.mean(dim=1)  # [B, D]
            else:
                return x.mean(dim=1)  # [B, D]
