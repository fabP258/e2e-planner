"""Trajectory visualization helpers for TensorBoard logging."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure


def plot_trajectory_comparison(
    image: torch.Tensor,
    pred_trajectory: torch.Tensor,
    gt_trajectory: torch.Tensor,
) -> Figure:
    """Plot camera frame alongside predicted vs ground-truth BEV trajectories.

    Args:
        image: Camera frame tensor [C, H, W] in [0, 1].
        pred_trajectory: Predicted trajectory [num_points, 2] (x=forward, y=lateral).
        gt_trajectory: Ground-truth trajectory [num_points, 2].

    Returns:
        Matplotlib Figure with two subplots.
    """
    fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(14, 5))

    # Camera frame
    ax_img.imshow(image.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
    ax_img.set_title("Reference frame")
    ax_img.axis("off")

    # BEV trajectory comparison
    gt = gt_trajectory.cpu().numpy()
    pred = pred_trajectory.cpu().numpy()

    ax_bev.plot(-gt[:, 1], gt[:, 0], "o-", color="tab:blue", markersize=4, label="ground truth")
    ax_bev.plot(-pred[:, 1], pred[:, 0], "x--", color="tab:orange", markersize=4, label="predicted")
    ax_bev.plot(0, 0, "s", color="tab:red", markersize=10, label="ego vehicle")
    ax_bev.set_xlabel("lateral [m]  (right \u2192 left)")
    ax_bev.set_ylabel("longitudinal [m]  (forward \u2191)")
    ax_bev.set_title("Future trajectory (bird's-eye view)")
    ax_bev.set_xlim(-5, 5)
    ax_bev.set_ylim(0, 40)
    ax_bev.legend()
    ax_bev.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
