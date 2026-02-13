"""Training script for E2E Planner.

Supports single-GPU and multi-GPU (DDP) training.

Single-GPU:
    python train.py --model-size medium ...

Multi-GPU (via torchrun):
    torchrun --nproc_per_node=2 train.py --model-size medium ...
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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


def _is_ddp() -> bool:
    """Check if launched via torchrun (DDP)."""
    return "LOCAL_RANK" in os.environ


def _setup_ddp() -> tuple[int, int]:
    """Initialize DDP process group. Returns (local_rank, world_size)."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def _cleanup_ddp():
    dist.destroy_process_group()


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model, unwrapping DDP if needed."""
    return model.module if isinstance(model, DDP) else model


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
    writer: SummaryWriter | None,
    global_step: int,
    train_config: TrainingConfig,
    rank: int = 0,
) -> tuple[float, int]:
    """Train for one epoch. Returns (avg_loss, global_step)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    loader = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True) if rank == 0 else dataloader
    for images, trajectories in loader:
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

        if rank == 0 and writer is not None:
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

        if rank == 0:
            loader.set_postfix(
                loss=f"{loss_value:.4f}",
                avg=f"{total_loss / num_batches:.4f}",
            )

    avg_loss = total_loss / num_batches

    # Average loss across all GPUs so the logged value reflects the full dataset
    if dist.is_initialized():
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    if rank == 0 and writer is not None:
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)

    return avg_loss, global_step


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: callable,
    device: torch.device,
    writer: SummaryWriter | None,
    epoch: int,
    train_config: TrainingConfig,
    rank: int = 0,
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
            if rank == 0 and writer is not None and not logged_images:
                _log_trajectory_images(
                    writer, "val", images, pred_trajectories,
                    trajectories, epoch, train_config.num_logged_images,
                )
                logged_images = True

    avg_loss = total_loss / num_batches

    if dist.is_initialized():
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    if rank == 0 and writer is not None:
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
        "model_state_dict": _unwrap_model(model).state_dict(),
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
    device: torch.device | None = None,
) -> tuple[int, int]:
    """Load training checkpoint. Returns (starting_epoch, global_step)."""
    checkpoint = torch.load(path, map_location=device)
    _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
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

    # --- DDP setup ---
    ddp = _is_ddp()
    if ddp:
        local_rank, world_size = _setup_ddp()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        if rank == 0:
            print(f"DDP: {world_size} GPUs, backend=nccl")
    else:
        rank = 0
        device = torch.device(args.device)

    if rank == 0:
        print(f"Using device: {device}")

    # Initialize configs
    model_config = getattr(ModelConfig, args.model_size)()
    train_config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size

    # Create model
    model = E2EPlanner.from_config(model_config)
    model = model.to(device)
    if rank == 0:
        print(f"Model parameters: {model.count_parameters():,}")

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # Create datasets
    if rank == 0:
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

    if rank == 0:
        print(f"Training samples: {len(train_dataset)}")

    # Create dataloaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )

    val_sampler = None
    val_loader = None
    if val_dataset:
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=train_config.num_workers,
            pin_memory=True,
        )
        if rank == 0:
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
        start_epoch, global_step = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint, device=device,
        )

    # Create checkpoint directory & TensorBoard writer (rank 0 only)
    writer = None
    if rank == 0:
        checkpoint_dir = Path(train_config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        run_name = f"{args.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=str(Path(train_config.log_dir) / run_name))

        # Log hyperparameters
        writer.add_text("config/model", f"```\n{model_config}\n```", 0)
        writer.add_text("config/training", f"```\n{train_config}\n```", 0)
        writer.add_text("config/args", f"```\n{args}\n```", 0)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(start_epoch, train_config.num_epochs):
        # Ensure each DDP process sees a different shard each epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{train_config.num_epochs}")
            print(f"{'='*60}")

        # Train
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch + 1,
            writer, global_step, train_config, rank,
        )
        if rank == 0:
            print(f"Training Loss: {train_loss:.6f}")

        # Validate
        if val_loader:
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            val_loss = validate(
                model, val_loader, loss_fn, device,
                writer, epoch + 1, train_config, rank,
            )
            if rank == 0:
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

        if rank == 0:
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
    if rank == 0:
        save_checkpoint(
            model, optimizer, scheduler, train_config.num_epochs - 1,
            train_loss, checkpoint_dir / "final_model.pt",
            global_step,
        )
        writer.close()
        print("\nTraining complete!")

    if ddp:
        _cleanup_ddp()


if __name__ == "__main__":
    main()
