"""MLP planning head for trajectory prediction."""

import torch
import torch.nn as nn


class PlanningHead(nn.Module):
    """
    MLP head that predicts trajectory points from fused temporal features.

    Takes encoded features and outputs a sequence of (x, y) trajectory points.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_trajectory_points: int = 18,
        num_hypotheses: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_points = num_trajectory_points
        self.num_hypotheses = num_hypotheses
        # for each hypotheses: weight + num_trajectory_points (x,y) points
        self.output_dim = num_hypotheses + num_hypotheses * num_trajectory_points * 2

        # MLP: Linear -> ReLU -> Linear -> ReLU -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Fused temporal features of shape [B, embed_dim]
               or [B, num_frames, embed_dim] (will be pooled)

        Returns:
            Logits of shape [B, num_hypotheses]
            Trajectory predictions of shape [B, num_hypotheses, num_points, 2]
        """
        # Handle sequence input by mean pooling
        if x.dim() == 3:
            x = x.mean(dim=1)  # [B, T, D] -> [B, D]

        # Apply MLP
        out = self.mlp(x)  # [B, num_hypotheses + num_hypotheses * num_points * 2]

        logits = out[:,:self.num_hypotheses] # [B, num_hypotheses]

        trajectory = out[:,self.num_hypotheses:] # [B, num_hypotheses * num_points * 2]
        # [B, num_hypotheses * num_points * 2] -> [B, num_hypotheses, num_points, 2]
        trajectory = trajectory.view(-1, self.num_hypotheses, self.num_points, 2)

        return logits, trajectory
