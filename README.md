# Finetune openpi using LoRA and Deploy

This repository contains a script for finetuning openpi using LoRA and deploying it to Hugging Face's Lerobot Repo.

Based on openpi's readme, pi0 model can be finetuned using LoRA on a single 4090 GPU.
In this repo, it includes all my implementation of this process.

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
pip install uv # if you don't have uv installed.
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

**Docker**: As an alternative to uv installation, we provide instructions for installing openpi using Docker. If you encounter issues with your system setup, consider using Docker to simplify installation. See [Docker Setup](docs/docker.md) for more details.

# Activate  the virtual environment
```bash
source .venv/bin/activate
pip install boto3 tqdm_loggable types-boto3-s3 # you may need to install these packages 
```

# Training Configuration
You need to add your training configuration for your task. 
You can check pi0_mycobot_low_mem_finetune TrainConfig in src/openpi/training/config.py

# Get Access for the Large Language Model openpi used and login to huggingface

# Training
Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_mycobot_low_mem_finetune
```
Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):
Or you can replace the `--overwrite` flag to `--resume` if you want to resume training from the last checkpoint.
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_mycobot_low_mem_finetune --exp-name=my_experiment --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_mycobot_low_mem_finetune --exp-name=my_experiment --resume
```

# Merge Lora Weights to Base Model
Cause we want to use Lerobot to load our new model weights and Lerobot repo only define the base model without any LoRA layer, we need to merge all LoRA weights to the base model.
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/merge_lora_layer.py pi0_mycobot_low_mem_finetune --exp-name=my_experiment --resume
```

After the process above, we finish all process on this repo and we can move to Lerobot to continue your work.


