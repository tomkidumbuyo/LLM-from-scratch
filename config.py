import os
import torch

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "the-verdict.txt"
MODEL_PATH = BASE_DIR / "checkpoints" / "gpt_model.pth"

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

TRAIN_CONFIG = {
    "batch_size": int(os.getenv("DATALOADER_BATCH_SIZE", 2)),
    "max_length": int(os.getenv("DATALOADER_MAX_LENGTH", 256)),
    "stride": int(os.getenv("DATALOADER_STRIDE", 256)),
    "shuffle": os.getenv("DATALOADER_SHUFFLE", "true").lower() == "true",
    "train_ratio": 0.90,
    "n_epochs": 15,
    "initial_lr": 1e-5,
    "peak_lr": 5e-4,
    "min_lr": 1e-5,
    "warmup_ratio": 0.20,
    "weight_decay": 0.1,
    "eval_freq": 5,
    "eval_iter": 1,
    "start_context": "Every effort moves you",
    "seed": 123
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")