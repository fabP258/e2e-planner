"""Configuration for the E2E Planner model."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Input configuration
    num_frames: int = 3
    image_height: int = 288
    image_width: int = 512
    in_channels: int = 3

    # Patch embedding
    patch_size: int = 16
    embed_dim: int = 256

    # Spatial transformer
    spatial_num_layers: int = 4
    spatial_num_heads: int = 8
    spatial_mlp_ratio: float = 4.0
    spatial_dropout: float = 0.1

    # Temporal transformer
    temporal_num_layers: int = 2
    temporal_num_heads: int = 8
    temporal_mlp_ratio: float = 4.0
    temporal_dropout: float = 0.1

    # Planning head
    planning_hidden_dim: int = 512
    num_trajectory_points: int = 40
    num_hypotheses: int = 5

    # CLS token for temporal attention
    use_cls_token: bool = True

    @classmethod
    def small(cls) -> "ModelConfig":
        """~5.4M params."""
        return cls(
            embed_dim=256, spatial_num_layers=4, spatial_num_heads=8,
            temporal_num_layers=2, temporal_num_heads=8, planning_hidden_dim=512,
        )

    @classmethod
    def medium(cls) -> "ModelConfig":
        """~40M params."""
        return cls(
            embed_dim=512, spatial_num_layers=8, spatial_num_heads=8,
            temporal_num_layers=4, temporal_num_heads=8, planning_hidden_dim=1024,
        )

    @classmethod
    def large(cls) -> "ModelConfig":
        """~132M params."""
        return cls(
            embed_dim=768, spatial_num_layers=12, spatial_num_heads=12,
            temporal_num_layers=6, temporal_num_heads=12, planning_hidden_dim=1536,
        )

    @property
    def num_patches_h(self) -> int:
        return self.image_height // self.patch_size

    @property
    def num_patches_w(self) -> int:
        return self.image_width // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.num_patches_h * self.num_patches_w


@dataclass
class TrainingConfig:
    # Data
    batch_size: int = 2
    num_workers: int = 8

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_epochs: int = 5

    # Loss weighting (optional: weight near-term points more)
    use_weighted_loss: bool = False
    near_term_weight: float = 2.0
    near_term_points: int = 6  # First 6 points get higher weight

    # Component-wise loss normalization (equalize x/y contribution)
    normalize_loss_components: bool = True
    trajectory_x_std: float = 8.4239
    trajectory_y_std: float = 0.6838

    # Multi-hypothesis loss balance
    regression_loss_weight: float = 1.0

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10

    # Logging
    log_dir: str = "runs"
    log_images_every: int = 1000   # log trajectory visualizations every N steps (0 = disabled)
    num_logged_images: int = 4    # samples per visualization

