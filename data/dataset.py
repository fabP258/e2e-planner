"""Dataset for loading driving image sequences and trajectories."""

import os
import json
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from physical_ai_av import PhysicalAIAVDatasetInterface
    from physical_ai_av import egomotion as _egomotion_module
    from physical_ai_av.video import SeekVideoReader
    import pandas as pd
    HAS_PHYSICAL_AI_AV = True
except ImportError:
    HAS_PHYSICAL_AI_AV = False


class DrivingSequenceDataset(Dataset):
    """
    Dataset for loading sequences of driving images and ground truth trajectories.

    Expected data structure:
    data_dir/
    ├── sequence_000/
    │   ├── frame_0.png
    │   ├── frame_1.png
    │   ├── ...
    │   ├── frame_5.png
    │   └── trajectory.json  # or trajectory.npy
    ├── sequence_001/
    │   └── ...

    trajectory.json format:
    {
        "points": [[x0, y0], [x1, y1], ..., [x17, y17]]
    }

    Or trajectory.npy: numpy array of shape [18, 2]
    """

    def __init__(
        self,
        data_dir: str,
        num_frames: int = 6,
        num_trajectory_points: int = 18,
        image_height: int = 128,
        image_width: int = 256,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory containing sequence folders
            num_frames: Number of frames per sequence
            num_trajectory_points: Number of trajectory points
            image_height: Target image height
            image_width: Target image width
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.num_trajectory_points = num_trajectory_points
        self.image_height = image_height
        self.image_width = image_width
        self.transform = transform

        # Find all sequence directories
        self.sequences = self._find_sequences()

    def _find_sequences(self) -> List[Path]:
        """Find all valid sequence directories."""
        sequences = []

        if not self.data_dir.exists():
            return sequences

        for seq_dir in sorted(self.data_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            # Check if sequence has required files
            has_frames = all(
                (seq_dir / f"frame_{i}.png").exists() or
                (seq_dir / f"frame_{i}.jpg").exists()
                for i in range(self.num_frames)
            )
            has_trajectory = (
                (seq_dir / "trajectory.json").exists() or
                (seq_dir / "trajectory.npy").exists()
            )

            if has_frames and has_trajectory:
                sequences.append(seq_dir)

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence of images and corresponding trajectory.

        Args:
            idx: Sequence index

        Returns:
            images: Tensor of shape [num_frames, C, H, W]
            trajectory: Tensor of shape [num_points, 2]
        """
        seq_dir = self.sequences[idx]

        # Load images
        images = self._load_images(seq_dir)

        # Load trajectory
        trajectory = self._load_trajectory(seq_dir)

        return images, trajectory

    def _load_images(self, seq_dir: Path) -> torch.Tensor:
        """Load and preprocess sequence of images."""
        if not HAS_PIL:
            raise ImportError("PIL is required for loading images. Install with: pip install pillow")

        frames = []
        for i in range(self.num_frames):
            # Try png first, then jpg
            img_path = seq_dir / f"frame_{i}.png"
            if not img_path.exists():
                img_path = seq_dir / f"frame_{i}.jpg"

            # Load image
            img = Image.open(img_path).convert("RGB")

            # Resize if needed
            if img.size != (self.image_width, self.image_height):
                img = img.resize(
                    (self.image_width, self.image_height),
                    Image.BILINEAR,
                )

            # Convert to tensor
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

            frames.append(img_tensor)

        # Stack frames: [num_frames, C, H, W]
        images = torch.stack(frames, dim=0)

        # Apply transform if provided
        if self.transform is not None:
            images = self.transform(images)

        return images

    def _load_trajectory(self, seq_dir: Path) -> torch.Tensor:
        """Load trajectory from file."""
        json_path = seq_dir / "trajectory.json"
        npy_path = seq_dir / "trajectory.npy"

        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
            points = np.array(data["points"], dtype=np.float32)
        elif npy_path.exists():
            points = np.load(npy_path).astype(np.float32)
        else:
            raise FileNotFoundError(f"No trajectory file found in {seq_dir}")

        # Ensure correct shape
        assert points.shape == (self.num_trajectory_points, 2), \
            f"Expected trajectory shape ({self.num_trajectory_points}, 2), got {points.shape}"

        return torch.from_numpy(points)


class ImageFolderSequenceDataset(Dataset):
    """
    Dataset for loading sequences from folders with consecutively numbered images.

    Loads real images but uses dummy trajectories (for pipeline testing).

    Recursively searches for image folders. Supports structures like:
    data_dir/
    ├── location_a/
    │   ├── 2024-01-01/
    │   │   └── images/
    │   │       ├── 0000.png
    │   │       ├── 0001.png
    │   │       └── ...
    │   └── 2024-01-02/
    │       └── images/
    │           └── ...
    └── location_b/
        └── ...

    Creates sliding windows of num_frames consecutive images.
    """

    def __init__(
        self,
        data_dir: str,
        num_frames: int = 6,
        num_trajectory_points: int = 18,
        image_height: int = 128,
        image_width: int = 256,
        stride: int = 1,
        image_folder_name: str = "images",
        transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory to recursively search for image folders
            num_frames: Number of frames per sequence
            num_trajectory_points: Number of trajectory points
            image_height: Target image height
            image_width: Target image width
            stride: Stride between consecutive sequences (1 = all windows, higher = skip)
            image_folder_name: Name of folders containing images (e.g., "images")
            transform: Optional transform to apply to images
        """
        if not HAS_PIL:
            raise ImportError("PIL is required. Install with: pip install pillow")

        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.num_trajectory_points = num_trajectory_points
        self.image_height = image_height
        self.image_width = image_width
        self.stride = stride
        self.image_folder_name = image_folder_name
        self.transform = transform

        # Find all valid sequences (folder, start_idx, image_files)
        self.samples = self._find_samples()
        print(f"Found {len(self.samples)} samples from {len(self._image_folders)} folders in {self.data_dir}")

    def _find_image_folders(self) -> List[Path]:
        """Recursively find all folders named image_folder_name."""
        image_folders = []

        for path in self.data_dir.rglob(self.image_folder_name):
            if path.is_dir():
                image_folders.append(path)

        return sorted(image_folders)

    def _find_samples(self) -> List[Tuple[Path, int, List[Path]]]:
        """Find all valid sliding window samples."""
        samples = []

        if not self.data_dir.exists():
            self._image_folders = []
            return samples

        # Find all image folders recursively
        self._image_folders = self._find_image_folders()

        for img_folder in self._image_folders:
            # Find all image files in this folder
            image_files = self._get_sorted_images(img_folder)

            if len(image_files) < self.num_frames:
                continue

            # Create sliding windows
            for start_idx in range(0, len(image_files) - self.num_frames + 1, self.stride):
                samples.append((img_folder, start_idx, image_files))

        return samples

    def _get_sorted_images(self, folder: Path) -> List[Path]:
        """Get sorted list of image files in a folder."""
        extensions = {'.png', '.jpg', '.jpeg'}
        images = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        # Sort by filename (assumes numeric naming like 0000.png, 0001.png)
        return sorted(images, key=lambda x: x.stem)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence of images and dummy trajectory.

        Args:
            idx: Sample index

        Returns:
            images: Tensor of shape [num_frames, C, H, W]
            trajectory: Tensor of shape [num_points, 2] (dummy values)
        """
        seq_dir, start_idx, image_files = self.samples[idx]

        # Load images
        frames = []
        for i in range(self.num_frames):
            img_path = image_files[start_idx + i]
            img = Image.open(img_path).convert("RGB")

            # Resize if needed
            if img.size != (self.image_width, self.image_height):
                img = img.resize(
                    (self.image_width, self.image_height),
                    Image.BILINEAR,
                )

            # Convert to tensor
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            frames.append(img_tensor)

        images = torch.stack(frames, dim=0)

        if self.transform is not None:
            images = self.transform(images)

        # Generate dummy trajectory (straight line forward)
        trajectory = torch.zeros(self.num_trajectory_points, 2)
        for i in range(self.num_trajectory_points):
            trajectory[i, 0] = (i + 1) * 0.5  # x: forward
            trajectory[i, 1] = 0.0            # y: no lateral motion

        return images, trajectory


class DummyDrivingDataset(Dataset):
    """
    Dummy dataset for testing and debugging.

    Generates random images and trajectories.
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_frames: int = 6,
        num_trajectory_points: int = 18,
        image_height: int = 128,
        image_width: int = 256,
        in_channels: int = 3,
    ):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.num_trajectory_points = num_trajectory_points
        self.image_height = image_height
        self.image_width = image_width
        self.in_channels = in_channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random images
        images = torch.randn(
            self.num_frames,
            self.in_channels,
            self.image_height,
            self.image_width,
        )

        # Generate smooth random trajectory
        # Start from origin and gradually move forward with slight randomness
        trajectory = torch.zeros(self.num_trajectory_points, 2)
        for i in range(self.num_trajectory_points):
            # x increases roughly linearly (forward motion)
            trajectory[i, 0] = (i + 1) * 0.5 + torch.randn(1).item() * 0.1
            # y has slight random drift (lateral motion)
            trajectory[i, 1] = torch.randn(1).item() * 0.3

        return images, trajectory
