"""End-to-end planner combining all components."""

import torch
import torch.nn as nn

from .patch_embed import PatchEmbedding
from .spatial_encoder import SpatialTransformerEncoder
from .temporal_encoder import TemporalTransformerEncoder
from .planning_head import PlanningHead


class E2EPlanner(nn.Module):
    """
    End-to-end driving policy network.

    Takes a sequence of RGB images and predicts trajectory points.

    Architecture:
    [6 RGB Images] → [Patch Embedding] → [Spatial Transformer] →
    [Temporal Transformer] → [MLP Head] → [18 (x,y) points]
    """

    def __init__(
        self,
        num_frames: int = 6,
        image_height: int = 128,
        image_width: int = 256,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 256,
        spatial_num_layers: int = 4,
        spatial_num_heads: int = 8,
        spatial_mlp_ratio: float = 4.0,
        spatial_dropout: float = 0.1,
        temporal_num_layers: int = 2,
        temporal_num_heads: int = 8,
        temporal_mlp_ratio: float = 4.0,
        temporal_dropout: float = 0.1,
        planning_hidden_dim: int = 512,
        num_trajectory_points: int = 18,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # Patch embedding (shared across all frames)
        self.patch_embed = PatchEmbedding(
            image_height=image_height,
            image_width=image_width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_cls_token=use_cls_token,
            dropout=spatial_dropout,
        )

        # Spatial transformer encoder
        self.spatial_encoder = SpatialTransformerEncoder(
            embed_dim=embed_dim,
            num_layers=spatial_num_layers,
            num_heads=spatial_num_heads,
            mlp_ratio=spatial_mlp_ratio,
            dropout=spatial_dropout,
        )

        # Temporal transformer encoder
        self.temporal_encoder = TemporalTransformerEncoder(
            embed_dim=embed_dim,
            num_layers=temporal_num_layers,
            num_heads=temporal_num_heads,
            mlp_ratio=temporal_mlp_ratio,
            dropout=temporal_dropout,
            num_frames=num_frames,
            use_cls_only=use_cls_token,
        )

        # Planning head
        self.planning_head = PlanningHead(
            embed_dim=embed_dim,
            hidden_dim=planning_hidden_dim,
            num_trajectory_points=num_trajectory_points,
            dropout=temporal_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images of shape [B, num_frames, C, H, W]

        Returns:
            Trajectory predictions of shape [B, num_points, 2]
        """
        B, T, C, H, W = x.shape
        assert T == self.num_frames, f"Expected {self.num_frames} frames, got {T}"

        # Process each frame through patch embedding
        # Reshape: [B, T, C, H, W] -> [B*T, C, H, W]
        x = x.view(B * T, C, H, W)

        # Patch embedding: [B*T, C, H, W] -> [B*T, num_patches, embed_dim]
        x = self.patch_embed(x)

        # Reshape back: [B*T, N, D] -> [B, T, N, D]
        N = x.shape[1]  # num_patches (+ 1 if CLS token)
        x = x.view(B, T, N, self.embed_dim)

        # Spatial encoding: [B, T, N, D] -> [B, T, N, D]
        x = self.spatial_encoder(x)

        # Temporal encoding: [B, T, N, D] -> [B, D]
        x = self.temporal_encoder(x, return_sequence=False)

        # Planning head: [B, D] -> [B, num_points, 2]
        trajectory = self.planning_head(x)

        return trajectory

    @classmethod
    def from_config(cls, config) -> "E2EPlanner":
        """Create model from config object."""
        return cls(
            num_frames=config.num_frames,
            image_height=config.image_height,
            image_width=config.image_width,
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            spatial_num_layers=config.spatial_num_layers,
            spatial_num_heads=config.spatial_num_heads,
            spatial_mlp_ratio=config.spatial_mlp_ratio,
            spatial_dropout=config.spatial_dropout,
            temporal_num_layers=config.temporal_num_layers,
            temporal_num_heads=config.temporal_num_heads,
            temporal_mlp_ratio=config.temporal_mlp_ratio,
            temporal_dropout=config.temporal_dropout,
            planning_hidden_dim=config.planning_hidden_dim,
            num_trajectory_points=config.num_trajectory_points,
            use_cls_token=config.use_cls_token,
        )

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
