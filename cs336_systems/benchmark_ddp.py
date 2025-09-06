
import os
import random
import timeit
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.nn_model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.data_loader import RandomStartBatchSampler, TokensDataset
from cs336_systems.common import *
from cs336_systems.ddp_wrapper import *

from logging import getLogger
logger = getLogger(__name__)

world_size = 2

warm_up_steps = 3
benchmark_steps = 3

config = MODEL_SIZE_CONFIGS["xl"]
context_length = 128
batch_size = 12


def benchmark_ddp(rank: int, world_size: int, wrapper_cls, *args):

    assert batch_size % world_size == 0, "Batch size must be divisible by world size"

    _setup_random_seed(42)

    device = _setup_process_group(rank=rank, world_size=world_size, backend="nccl")
    dist.barrier(device_ids=[rank])

    shard_size = batch_size // world_size
    shard_offset = rank * shard_size

    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        rope_theta=10000.0,
        flash_attn=False,
        device=device
    )
    wrapper = wrapper_cls(model, *args)
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
    model.to(device)

    total_t1_t2, total_t2_t3, total_t3_t4 = torch.tensor(0.0, device=device), torch.tensor(0.0,  device=device), torch.tensor(0.0,  device=device)

    if rank == 0:
        logger.info("Starting benchmark...")

    with torch.amp.autocast(device_type="cuda"):
        for i, (inputs, targets) in enumerate(train_loader):

            optimizer.zero_grad()

            if i < warm_up_steps:
                nvtx_range_name = f"Warmup {i}"
            else:
                if i == warm_up_steps:
                    pass
                nvtx_range_name = f"Iteration {i - warm_up_steps}"

            with torch.cuda.nvtx.range(nvtx_range_name):
                with torch.cuda.nvtx.range(f"data to GPU"):
                    inputs = inputs[shard_offset:shard_offset + shard_size].to(device, non_blocking=True)
                    targets = targets[shard_offset:shard_offset + shard_size].to(device, non_blocking=True)

                t1 = timeit.default_timer()

                with torch.cuda.nvtx.range(f"forward pass"):
                    outputs = wrapper(inputs)
                    torch.cuda.synchronize()

                t2 = timeit.default_timer()

                with torch.cuda.nvtx.range(f"loss computation"):
                    loss = cross_entropy(outputs, targets, reduce=True)
                    torch.cuda.synchronize()

                t3 = timeit.default_timer()

                with torch.cuda.nvtx.range(f"backward pass"):
                    scaler.scale(loss).backward()
                    wrapper.finish_gradient_synchronization()
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.synchronize()

                t4 = timeit.default_timer()

            if i < warm_up_steps:
                continue
            
            if rank == 0:
                logger.info(f"Step {i - warm_up_steps + 1}:")
                logger.info(f"  Forward pass: {t2 - t1:.6f} seconds")
                logger.info(f"  Loss computation: {t3 - t2:.6f} seconds")
                logger.info(f"  Backward pass: {t4 - t3:.6f} seconds")

            total_t1_t2 += t2 - t1
            total_t2_t3 += t3 - t2
            total_t3_t4 += t4 - t3

            if i == warm_up_steps + benchmark_steps - 1:
                dist.reduce(total_t1_t2, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(total_t2_t3, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(total_t3_t4, dst=0, op=dist.ReduceOp.SUM)
                avg_t1_t2 = total_t1_t2 / benchmark_steps / world_size
                avg_t2_t3 = total_t2_t3 / benchmark_steps / world_size
                avg_t3_t4 = total_t3_t4 / benchmark_steps / world_size
                if rank == 0:
                    logger.info(f"Average times over {benchmark_steps} steps and {world_size} processes:")
                    logger.info(f"  Forward pass: {avg_t1_t2.item():.6f} seconds")
                    logger.info(f"  Loss computation: {avg_t2_t3.item():.6f} seconds")
                    logger.info(f"  Backward pass: {avg_t3_t4.item():.6f} seconds")
                    logger.info(f"  Total: {avg_t1_t2.item() + avg_t2_t3.item() + avg_t3_t4.item():.6f} seconds")
                break

    _cleanup_process_group()


def _setup_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    # https://discuss.pytorch.org/t/should-local-rank-be-equal-to-torch-cuda-current-device/150873/2
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def _cleanup_process_group():
    # Synchronize before we destroy the process group
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    for bucket in [1, 10, 100, 1000, 5000]:
        mp.spawn(
            benchmark_ddp,
            args=(world_size, DDPOverlapBucketedWrapper, bucket),
            nprocs=world_size,
            join=True,
        )