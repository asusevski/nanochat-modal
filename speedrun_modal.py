"""
Modal speedrun script for nanochat training.

This script replicates the functionality of speedrun.sh but runs on Modal infrastructure.
It builds a complete training environment including Rust toolchain and rustbpe tokenizer.

=== PREREQUISITES ===

1. Install Modal:
   pip install modal

2. Set up Modal account and authenticate:
   modal setup

3. (Optional) Configure wandb for logging:
   - Create a Modal secret with your wandb API key:
     modal secret create wandb-secret WANDB_API_KEY=your_key_here
   - Uncomment the secrets line in the @app.function decorator

=== USAGE ===

1. Basic usage (no wandb logging):
   modal run speedrun_modal.py

2. With wandb logging:
   modal run speedrun_modal.py --wandb-run=my-run-name

3. Deploy as a persistent app:
   modal deploy speedrun_modal.py

=== WHAT IT DOES ===

This script performs the complete nanochat training pipeline on 8xH100 GPUs:
1. Resets training reports
2. Downloads dataset shards (~24GB total)
3. Trains a BPE tokenizer on 2B characters
4. Pretrains a 561M parameter transformer model (d20)
5. Evaluates the base model on CORE tasks
6. Runs midtraining to teach conversation and tool use
7. Performs supervised finetuning
8. Generates a comprehensive training report

Expected runtime: ~4 hours on 8xH100 GPUs
Expected cost: ~$100 (8 GPUs × $3/hr × 4 hours)

All training artifacts (datasets, checkpoints, reports) are saved to a persistent
Modal volume and can be reused across runs.

=== ARCHITECTURE ===

The Modal image includes:
- Debian slim base with Python 3.11
- Rust toolchain (cargo, rustc) for rustbpe tokenizer
- uv package manager for fast dependency installation
- PyTorch with CUDA 12.8 support
- All nanochat Python dependencies
- Pre-built rustbpe tokenizer (Rust+Python via maturin)
"""

import modal

# Create Modal app
app = modal.App("nanochat-speedrun")

# Create a volume for persistent storage of datasets and model checkpoints
volume = modal.Volume.from_name("nanochat-data", create_if_missing=True)

# Build the Modal image with all dependencies
def build_nanochat_image():
    """
    Builds a Modal image that contains:
    - Rust toolchain (cargo, rustc)
    - Python dependencies from pyproject.toml
    - rustbpe tokenizer built with maturin
    """

    # Start with a CUDA-enabled base image with PyTorch
    # Using debian_slim and will install CUDA via pip packages
    image = (
        modal.Image.debian_slim(python_version="3.11")
        # Install system dependencies
        .apt_install(
            "curl",
            "build-essential",
            "git",
            "ca-certificates",
            "pkg-config",
            "libssl-dev",
        )
        # Install Rust toolchain
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            "echo 'source $HOME/.cargo/env' >> ~/.bashrc",
        )
        # Set environment variables for Rust
        .env({"PATH": "/root/.cargo/bin:$PATH"})
        .env({"CARGO_HOME": "/root/.cargo"})
        .env({"RUSTUP_HOME": "/root/.rustup"})
        # Install uv for Python package management
        .run_commands(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "echo 'source $HOME/.local/bin/env' >> ~/.bashrc",
        )
        .env({"PATH": "/root/.local/bin:$PATH"})
        # Copy project files (exclude .venv, .git, and cache directories)
        .copy_local_dir(
            ".",
            "/root/nanochat",
            exclude=[".venv", ".git", "__pycache__", "*.pyc", ".pytest_cache"]
        )
        .workdir("/root/nanochat")
        # Install Python dependencies using uv with GPU support
        .run_commands(
            "cd /root/nanochat && /root/.local/bin/uv sync --extra gpu",
        )
        # Build the rustbpe tokenizer with maturin
        .run_commands(
            "cd /root/nanochat && export PATH=/root/.cargo/bin:$PATH && /root/.local/bin/uv run maturin develop --release --manifest-path rustbpe/Cargo.toml",
        )
        # Set OMP_NUM_THREADS for performance
        .env({"OMP_NUM_THREADS": "1"})
    )

    return image

# Create the image
nanochat_image = build_nanochat_image()


@app.function(
    image=nanochat_image,
    gpu=modal.gpu.H100(count=8),  # 8xH100 GPUs as used in speedrun.sh
    timeout=60 * 60 * 5,  # 5 hours timeout (speedrun takes ~4 hours)
    volumes={"/root/.cache": volume},  # Mount volume for persistent data
    # Uncomment the line below if you have a wandb secret configured:
    # secrets=[modal.Secret.from_name("wandb-secret")],
)
def speedrun_training(wandb_run: str = "dummy"):
    """
    Main training function that replicates speedrun.sh workflow.

    Args:
        wandb_run: Name for wandb run (default: "dummy" to skip wandb logging)
    """
    import subprocess
    import os

    # Set environment variables
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NANOCHAT_BASE_DIR"] = "/root/.cache/nanochat"
    os.makedirs(os.environ["NANOCHAT_BASE_DIR"], exist_ok=True)

    # Number of GPUs (processes) to use
    nproc_per_node = 8

    print("=" * 80)
    print("Starting nanochat speedrun training on Modal")
    print("=" * 80)

    # 1. Reset report
    print("\n[1/9] Resetting report...")
    subprocess.run(["python", "-m", "nanochat.report", "reset"], check=True)

    # 2. Download initial dataset shards
    print("\n[2/9] Downloading initial 8 data shards (~800MB)...")
    subprocess.run(["python", "-m", "nanochat.dataset", "-n", "8"], check=True)

    # 3. Start downloading full dataset in background
    print("\n[3/9] Starting background download of 240 data shards (~24GB)...")
    dataset_proc = subprocess.Popen(["python", "-m", "nanochat.dataset", "-n", "240"])

    # 4. Train tokenizer
    print("\n[4/9] Training tokenizer on 2B characters...")
    subprocess.run(
        ["python", "-m", "scripts.tok_train", "--max_chars=2000000000"],
        check=True
    )

    # 5. Evaluate tokenizer
    print("\n[5/9] Evaluating tokenizer...")
    subprocess.run(["python", "-m", "scripts.tok_eval"], check=True)

    # 6. Wait for dataset download
    print("\n[6/9] Waiting for dataset download to complete...")
    dataset_proc.wait()

    # 7. Base model pretraining
    print("\n[7/9] Pretraining d20 base model (561M params)...")
    subprocess.run([
        "torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}",
        "-m", "scripts.base_train", "--",
        "--depth=20", f"--run={wandb_run}"
    ], check=True)

    print("\n[7/9] Evaluating base model loss...")
    subprocess.run([
        "torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}",
        "-m", "scripts.base_loss"
    ], check=True)

    print("\n[7/9] Evaluating base model on CORE tasks...")
    subprocess.run([
        "torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}",
        "-m", "scripts.base_eval"
    ], check=True)

    # 8. Midtraining
    print("\n[8/9] Downloading identity conversations...")
    subprocess.run([
        "curl", "-L", "-o",
        f"{os.environ['NANOCHAT_BASE_DIR']}/identity_conversations.jsonl",
        "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
    ], check=True)

    print("\n[8/9] Running midtraining...")
    subprocess.run([
        "torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}",
        "-m", "scripts.mid_train", "--", f"--run={wandb_run}"
    ], check=True)

    print("\n[8/9] Evaluating after midtraining...")
    subprocess.run([
        "torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}",
        "-m", "scripts.chat_eval", "--", "-i", "mid"
    ], check=True)

    # 9. Supervised Finetuning
    print("\n[9/9] Running supervised finetuning...")
    subprocess.run([
        "torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}",
        "-m", "scripts.chat_sft", "--", f"--run={wandb_run}"
    ], check=True)

    print("\n[9/9] Final evaluation after SFT...")
    subprocess.run([
        "torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}",
        "-m", "scripts.chat_eval", "--", "-i", "sft"
    ], check=True)

    # 10. Generate final report
    print("\n[10/10] Generating final report...")
    subprocess.run(["python", "-m", "nanochat.report", "generate"], check=True)

    # Commit volume to persist all data
    volume.commit()

    print("\n" + "=" * 80)
    print("Speedrun training complete!")
    print(f"Model checkpoints saved to Modal volume at: /root/.cache/nanochat")
    print("=" * 80)

    # Return path to the final model
    return {
        "status": "success",
        "cache_dir": os.environ["NANOCHAT_BASE_DIR"],
        "wandb_run": wandb_run,
    }


@app.local_entrypoint()
def main(wandb_run: str = "dummy"):
    """
    Local entrypoint to launch the speedrun training.

    Usage:
        modal run speedrun_modal.py
        modal run speedrun_modal.py --wandb-run=my-run-name
    """
    result = speedrun_training.remote(wandb_run=wandb_run)
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Results: {result}")
    print("=" * 80)
