
import timeit
from cs336_basics.optimizer import AdamW
import numpy as np
import torch
from torch.utils.data import DataLoader

from cs336_basics.nn_model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data_loader import RandomStartBatchSampler, TokensDataset
from cs336_systems.common import *

from logging import getLogger
logger = getLogger(__name__)

warm_up_steps = 3
benchmark_steps = 3

config = MODEL_SIZE_CONFIGS["small"]
context_length = 256

model = TransformerLM(
    vocab_size=VOCAB_SIZE,
    context_length=context_length,
    d_model=config.d_model,
    d_ff=config.d_ff,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    rope_theta=10000.0,
    device=DEVICE
)
train_ds = TokensDataset(data=np.load("../assignment1-basics/artifacts/tokens_tinystoriesV2_valid_train:v0/tokens.npy"), context_length=context_length)
train_loader = DataLoader(
    train_ds,
    batch_sampler=RandomStartBatchSampler(len(train_ds), batch_size=BATCH_SIZE),
    pin_memory=True,
    num_workers=8,
    persistent_workers=True,
)
optimizer = AdamW(model.parameters())
scaler = torch.amp.GradScaler()

model.train()
model.to(DEVICE)

total_t1_t2, total_t2_t3, total_t3_t4 = 0.0, 0.0, 0.0

logger.info("Starting benchmark...")

with torch.amp.autocast(dtype=torch.float32, device_type="cuda"):
    for i, (inputs, targets) in enumerate(train_loader):

        optimizer.zero_grad()

        if i < warm_up_steps:
            nvtx_range_name = f"Warmup {i}"
        else:
            if i == warm_up_steps:
                torch.cuda.memory._record_memory_history(max_entries=1000000)
            nvtx_range_name = f"Iteration {i - warm_up_steps}"

        with torch.cuda.nvtx.range(nvtx_range_name):
            with torch.cuda.nvtx.range(f"data to GPU"):
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

            t1 = timeit.default_timer()

            with torch.cuda.nvtx.range(f"forward pass"):
                outputs = model(inputs)
                torch.cuda.synchronize()

            t2 = timeit.default_timer()

            with torch.cuda.nvtx.range(f"loss computation"):
                loss = cross_entropy(outputs, targets, reduce=True)
                torch.cuda.synchronize()

            t3 = timeit.default_timer()

            with torch.cuda.nvtx.range(f"backward pass"):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()

            t4 = timeit.default_timer()

        if i < warm_up_steps:
            continue

        logger.info(f"Step {i - warm_up_steps + 1}:")
        logger.info(f"  Forward pass: {t2 - t1:.6f} seconds")
        logger.info(f"  Loss computation: {t3 - t2:.6f} seconds")
        logger.info(f"  Backward pass: {t4 - t3:.6f} seconds")

        total_t1_t2 += t2 - t1
        total_t2_t3 += t3 - t2
        total_t3_t4 += t4 - t3

        if i == warm_up_steps + benchmark_steps - 1:
            avg_t1_t2 = total_t1_t2 / benchmark_steps
            avg_t2_t3 = total_t2_t3 / benchmark_steps
            avg_t3_t4 = total_t3_t4 / benchmark_steps
            logger.info(f"Average times over {benchmark_steps} steps:")
            logger.info(f"  Forward pass: {avg_t1_t2:.6f} seconds")
            logger.info(f"  Loss computation: {avg_t2_t3:.6f} seconds")
            logger.info(f"  Backward pass: {avg_t3_t4:.6f} seconds")
            break

torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)