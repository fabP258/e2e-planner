"""Training script for E2E Planner."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import ModelConfig, TrainingConfig
from models import E2EPlanner
from data import NvidiaDriveDataset

try:
    from viz import plot_trajectory_comparison
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False


def _log_trajectory_images(
    writer: SummaryWriter,
    tag_prefix: str,
    images: torch.Tensor,
    pred_trajectories: torch.Tensor,
    gt_trajectories: torch.Tensor,
    global_step: int,
    num_images: int,
):
    """Log trajectory comparison figures to TensorBoard."""
    if not _HAS_VIZ:
        return
    n = min(num_images, images.shape[0])
    for i in range(n):
        # Use the last frame from the image stack
        last_frame = images[i, -1]  # [C, H, W]
        fig = plot_trajectory_comparison(last_frame, pred_trajectories[i], gt_trajectories[i])
        writer.add_figure(f"{tag_prefix}/trajectory_{i}", fig, global_step)


def create_weighted_loss(config: TrainingConfig):
    """Create weighted MSE loss that emphasizes near-term trajectory points."""

    def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Weighted MSE loss.

        Args:
            pred: Predicted trajectory [B, num_points, 2]
            target: Ground truth trajectory [B, num_points, 2]

        Returns:
            Scalar loss value
        """
        # Compute per-point MSE
        mse = (pred - target) ** 2  # [B, num_points, 2]

        if config.use_weighted_loss:
            # Create weights: higher for near-term points
            num_points = pred.shape[1]
            weights = torch.ones(num_points, device=pred.device)
            weights[:config.near_term_points] = config.near_term_weight

            # Normalize weights
            weights = weights / weights.sum() * num_points

            # Apply weights
            mse = mse * weights.view(1, -1, 1)

        return mse.mean()

    return weighted_mse_loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: callable,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
    train_config: TrainingConfig,
) -> tuple[float, int]:
    """Train for one epoch. Returns (avg_loss, global_step)."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_data_time = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    batch_start = time.perf_counter()
    for images, trajectories in pbar:
        data_time = time.perf_counter() - batch_start
        total_data_time += data_time

        images = images.to(device)
        trajectories = trajectories.to(device)

        # Forward pass
        optimizer.zero_grad()
        pred_trajectories = model(images)

        # Compute loss
        loss = loss_fn(pred_trajectories, trajectories)

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_value = loss.detach().item()
        total_loss += loss_value
        num_batches += 1

        # TensorBoard: per-step loss
        writer.add_scalar("train/loss_step", loss_value, global_step)

        # TensorBoard: trajectory visualizations
        if (
            train_config.log_images_every > 0
            and global_step % train_config.log_images_every == 0
        ):
            _log_trajectory_images(
                writer, "train", images, pred_trajectories.detach(),
                trajectories, global_step, train_config.num_logged_images,
            )

        global_step += 1

        avg_data_time = total_data_time / num_batches
        pbar.set_postfix(
            loss=f"{loss_value:.4f}",
            avg=f"{total_loss / num_batches:.4f}",
            data=f"{avg_data_time:.2f}s",  # TODO: remove after profiling
        )

        batch_start = time.perf_counter()

    avg_loss = total_loss / num_batches
    avg_data_time = total_data_time / num_batches

    # TensorBoard: epoch-level scalars
    writer.add_scalar("train/loss_epoch", avg_loss, epoch)
    writer.add_scalar("train/data_time_avg", avg_data_time, epoch)

    return avg_loss, global_step


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: callable,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    train_config: TrainingConfig,
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    logged_images = False

    with torch.no_grad():
        for images, trajectories in dataloader:
            images = images.to(device)
            trajectories = trajectories.to(device)

            pred_trajectories = model(images)
            loss = loss_fn(pred_trajectories, trajectories)

            total_loss += loss.item()
            num_batches += 1

            # Log trajectory visualizations from first batch
            if not logged_images:
                _log_trajectory_images(
                    writer, "val", images, pred_trajectories,
                    trajectories, epoch, train_config.num_logged_images,
                )
                logged_images = True

    avg_loss = total_loss / num_batches
    writer.add_scalar("val/loss_epoch", avg_loss, epoch)
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
    path: str,
    global_step: int = 0,
):
    """Save training checkpoint."""
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: str,
) -> tuple[int, int]:
    """Load training checkpoint. Returns (starting_epoch, global_step)."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    global_step = checkpoint.get("global_step", 0)
    print(f"Loaded checkpoint from {path}, epoch {checkpoint['epoch']}")
    return checkpoint["epoch"] + 1, global_step


def main():
    parser = argparse.ArgumentParser(description="Train E2E Planner")
    parser.add_argument("--chunk-ids", type=int, nargs="+", default=[0],
                        help="Chunk IDs to use (default: [0])")
    parser.add_argument("--val-chunk-ids", type=int, nargs="+", default=None,
                        help="Chunk IDs for validation")
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Model size preset (default: small)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: from TrainingConfig)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Initialize configs
    model_config = getattr(ModelConfig, args.model_size)()
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create model
    model = E2EPlanner.from_config(model_config)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create datasets
    print(f"Loading NVIDIA Drive dataset (chunks {args.chunk_ids})")
    train_dataset = NvidiaDriveDataset(
        chunk_ids=args.chunk_ids,
        num_frames=model_config.num_frames,
        num_trajectory_points=model_config.num_trajectory_points,
        image_height=model_config.image_height,
        image_width=model_config.image_width,
    )
    val_dataset = None
    if args.val_chunk_ids:
        val_dataset = NvidiaDriveDataset(
            chunk_ids=args.val_chunk_ids,
            num_frames=model_config.num_frames,
            num_trajectory_points=model_config.num_trajectory_points,
            image_height=model_config.image_height,
            image_width=model_config.image_width,
        )

    print(f"Training samples: {len(train_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=True,
        )
        print(f"Validation samples: {len(val_dataset)}")

    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_config.num_epochs,
        eta_min=train_config.learning_rate * 0.01,
    )

    # Loss function
    loss_fn = create_weighted_loss(train_config)

    # Load checkpoint if provided
    start_epoch = 0
    global_step = 0
    if args.checkpoint:
        start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, args.checkpoint)

    # Create checkpoint directory
    checkpoint_dir = Path(train_config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # TensorBoard writer
    run_name = f"{args.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(Path(train_config.log_dir) / run_name))

    # Log hyperparameters
    writer.add_text("config/model", f"```\n{model_config}\n```", 0)
    writer.add_text("config/training", f"```\n{train_config}\n```", 0)
    writer.add_text("config/args", f"```\n{args}\n```", 0)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(start_epoch, train_config.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{train_config.num_epochs}")
        print(f"{'='*60}")

        # Train
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch + 1,
            writer, global_step, train_config,
        )
        print(f"Training Loss: {train_loss:.6f}")

        # Validate
        if val_loader:
            val_loss = validate(
                model, val_loader, loss_fn, device,
                writer, epoch + 1, train_config,
            )
            print(f"Validation Loss: {val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_loss, checkpoint_dir / "best_model.pt",
                    global_step,
                )

        # Update scheduler
        scheduler.step()

        # TensorBoard: learning rate
        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], epoch + 1)

        # Save periodic checkpoint
        if (epoch + 1) % train_config.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                global_step,
            )

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, train_config.num_epochs - 1,
        train_loss, checkpoint_dir / "final_model.pt",
        global_step,
    )

    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
