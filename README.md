# E2E Planner

End-to-end motion planner for autonomous vehicles.

## Setup

Create a conda environment and install the package:

```bash
conda create -n e2e-planner python=3.12 -y
conda activate e2e-planner
pip install -e ".[dev]"
```

This installs all required dependencies including the
[NVIDIA PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
dataset SDK. The `dev` extra adds `matplotlib` and `tensorboard` for visualization and experiment tracking.

## Dataset

The `NvidiaDriveDataset` class handles downloading and loading data from the
NVIDIA PhysicalAI-Autonomous-Vehicles dataset. Specify which chunks to use via
`chunk_ids` -- data is downloaded automatically on first use and cached by the
HuggingFace Hub.

To verify the dataset setup:

```bash
python test_dataset.py
```

This loads chunk 0 and saves sample visualizations to `test_dataset_samples/`.

## Training

```bash
python train.py --nvidia-drive --chunk-ids 0 1 2
```

See `python train.py --help` for all options.

## Experiment Tracking

Training logs scalars, trajectory visualizations, and hyperparameters to
TensorBoard automatically. Event files are written to `runs/` by default.

```bash
tensorboard --logdir runs
```

Open `http://localhost:6006` to view:

- **Scalars**: `train/loss_step`, `train/loss_epoch`, `val/loss_epoch`, `train/learning_rate`, `train/data_time_avg`
- **Images**: predicted vs ground-truth trajectory plots (every 500 steps for training, every epoch for validation)
- **Text**: model, training, and CLI configs

Visualization frequency is controlled by `TrainingConfig.log_images_every` (default 500, set to 0 to disable).

## Known dependency constraints

**pandas < 3**: The `physical_ai_av` library's `EgomotionState.from_egomotion_df`
calls `.to_numpy()` on DataFrames read from parquet files without passing
`copy=True`. In pandas 3.0, Copy-on-Write is enabled by default, which causes
`.to_numpy()` to return read-only arrays. scipy's `Rotation.from_quat` then
fails with `buffer source array is read-only`. Until the upstream library is
fixed, pandas must be kept at 2.x.
