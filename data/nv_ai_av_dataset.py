from PIL import Image
import numpy as np
from typing import Optional, List, Tuple, Callable

import torch
from torch.utils.data import Dataset

from physical_ai_av import PhysicalAIAVDatasetInterface



class NvidiaDriveDataset(Dataset):
    """
    Dataset for the NVIDIA PhysicalAI-Autonomous-Vehicles dataset.

    Loads front camera frames and ego motion trajectories from downloaded
    chunks via the physical_ai_av package. Each sample consists of:
    - images: [num_frames, 3, H, W] tensor of video frames
    - trajectory: [num_trajectory_points, 2] tensor of future (x, y) waypoints
      in the ego vehicle's local coordinate frame (X=forward, Y=left)

    Note: Each __getitem__ call extracts data from zip archives on the fly.
    For better throughput, increase num_workers in the DataLoader or consider
    extracting the zip files to disk.
    """

    def __init__(
        self,
        chunk_ids: List[int],
        num_frames: int = 3,
        num_trajectory_points: int = 40,
        image_height: int = 128,
        image_width: int = 256,
        frame_skip: int = 0,
        trajectory_dt: float = 0.05,
        sample_stride: int = 3,
        camera: str = "camera_front_wide_120fov",
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            chunk_ids: Which chunks to use. Downloads if not already cached.
            num_frames: Number of input frames per sample
            num_trajectory_points: Number of future trajectory waypoints
            image_height: Target image height after resize
            image_width: Target image width after resize
            frame_skip: Video frames to skip between consecutive input frames.
                At 30fps, frame_skip=5 means ~0.17s between input frames.
            trajectory_dt: Time interval between trajectory waypoints in seconds
            sample_stride: Sliding window stride in video frames for generating samples
            camera: Camera feature name to use
            cache_dir: Optional HF Hub cache directory. Defaults to ~/.cache/huggingface/hub/
            transform: Optional transform to apply to images
        """

        self.num_frames = num_frames
        self.num_trajectory_points = num_trajectory_points
        self.image_height = image_height
        self.image_width = image_width
        self.frame_skip = frame_skip
        self.trajectory_dt = trajectory_dt
        self.sample_stride = sample_stride
        self.camera = camera
        self.transform = transform

        # Initialize dataset interface (downloads small metadata files if not cached)
        self._ds = PhysicalAIAVDatasetInterface(
            cache_dir=cache_dir,
            confirm_download_threshold_gb=float("inf"),
        )

        # Download chunk data (skips already-cached files)
        self._ds.download_chunk_features(
            chunk_ids, features=[self.camera, "egomotion"]
        )

        clip_ids = self._get_clip_ids_for_chunks(chunk_ids)

        # Pre-index: store timestamps per clip and build sample list
        self.clip_timestamps: dict = {}
        self.samples: List[Tuple[str, List[int]]] = []
        self._index_clips(clip_ids)

        print(
            f"NvidiaDriveDataset: {len(self.samples)} samples "
            f"from {len(self.clip_timestamps)} clips "
            f"(chunks {chunk_ids})"
        )

    def _get_clip_ids_for_chunks(self, chunk_ids: List[int]) -> List[str]:
        """Get clip IDs belonging to the specified chunks."""
        mask = self._ds.clip_index["chunk"].isin(chunk_ids)
        return self._ds.clip_index[mask].index.tolist()

    def _index_clips(self, clip_ids: List[str]):
        """Open each clip once to get frame timestamps and build sample index."""
        input_span = self.frame_skip * (self.num_frames - 1)
        future_duration_us = int(self.trajectory_dt * (self.num_trajectory_points - 1) * 1e6)

        for clip_id in clip_ids:
            try:
                # Get video frame timestamps
                video = self._ds.get_clip_feature(clip_id, self.camera)
                timestamps = video.timestamps.copy()
                video.close()

                # Get ego motion time range
                ego = self._ds.get_clip_feature(clip_id, "egomotion")
                ego_t0, ego_t1 = ego.time_range

                self.clip_timestamps[clip_id] = timestamps

                # Create sliding window samples
                for start_idx in range(0, len(timestamps) - input_span, self.sample_stride):
                    frame_indices = [
                        start_idx + i * self.frame_skip
                        for i in range(self.num_frames)
                    ]

                    if frame_indices[-1] >= len(timestamps):
                        break

                    # Ensure ego motion covers input frames and future trajectory
                    last_frame_ts = int(timestamps[frame_indices[-1]])
                    last_future_ts = last_frame_ts + future_duration_us

                    if last_frame_ts >= ego_t0 and last_future_ts <= ego_t1:
                        self.samples.append((clip_id, frame_indices))
            except Exception as e:
                print(f"Warning: skipping clip {clip_id}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample of video frames and future trajectory.

        Returns:
            images: Tensor of shape [num_frames, C, H, W]
            trajectory: Tensor of shape [num_trajectory_points, 2]
        """
        clip_id, frame_indices = self.samples[idx]
        timestamps = self.clip_timestamps[clip_id]

        images = self._load_frames(clip_id, frame_indices)
        trajectory = self._extract_trajectory(clip_id, timestamps, frame_indices)

        return images, trajectory

    def _load_frames(self, clip_id: str, frame_indices: List[int]) -> torch.Tensor:
        """Decode video frames, resize, and normalize to [0, 1]."""
        video = self._ds.get_clip_feature(clip_id, self.camera)
        indices_array = np.array(frame_indices, dtype=np.int64)
        raw_frames = video.decode_images_from_frame_indices(indices_array)
        video.close()

        # raw_frames: (N, H, W, 3) uint8
        frames = []
        for i in range(len(frame_indices)):
            img = Image.fromarray(raw_frames[i])
            if img.size != (self.image_width, self.image_height):
                img = img.resize(
                    (self.image_width, self.image_height),
                    Image.BILINEAR,
                )

            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [C, H, W]
            frames.append(img_tensor)

        images = torch.stack(frames, dim=0)  # [num_frames, C, H, W]

        if self.transform is not None:
            images = self.transform(images)

        return images

    def _extract_trajectory(
        self, clip_id: str, timestamps: np.ndarray, frame_indices: List[int]
    ) -> torch.Tensor:
        """Extract future trajectory waypoints in the ego vehicle's local frame."""
        ego = self._ds.get_clip_feature(clip_id, "egomotion")

        # Reference timestamp: last input frame
        ref_ts = int(timestamps[frame_indices[-1]])
        dt_us = int(self.trajectory_dt * 1e6)

        # Build timestamps: [ref, future_1, future_2, ..., future_{N-1}]
        # The reference point (origin) counts as the first of num_trajectory_points
        num_future = self.num_trajectory_points - 1
        all_ts = np.array(
            [ref_ts] + [ref_ts + (i + 1) * dt_us for i in range(num_future)]
        )

        # Batch-interpolate all poses at once
        all_states = ego(all_ts)
        all_pos = all_states.pose.translation  # (N+1, 3)
        all_rot = all_states.pose.rotation     # Rotation with N+1 elements

        # Transform all positions (including reference) into the reference frame's local coordinates
        ref_pos = all_pos[0]            # (3,)
        ref_rot_inv = all_rot[0].inv()  # single Rotation

        delta = all_pos - ref_pos       # (N+1, 3)
        local = ref_rot_inv.apply(delta)  # (N+1, 3)

        # Take (x=forward, y=left) and discard z
        # First point is the reference frame origin (0, 0)
        trajectory = torch.from_numpy(local[:, :2].astype(np.float32))

        return trajectory