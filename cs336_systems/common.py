from dataclasses import dataclass

import torch

@dataclass
class ModelSizeConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int

MODEL_SIZE_CONFIGS = {
    "small": ModelSizeConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelSizeConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelSizeConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelSizeConfig(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7b": ModelSizeConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

VOCAB_SIZE = 10000
BATCH_SIZE = 4
DEVICE = torch.device("cuda")