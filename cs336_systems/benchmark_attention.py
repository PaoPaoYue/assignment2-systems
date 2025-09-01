
from multiprocessing import context
import timeit
import numpy as np
import torch
from torch.utils.data import DataLoader

from cs336_basics.nn_model import MultiheadSelfAttention, TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data_loader import RandomStartBatchSampler, TokensDataset
from cs336_basics.optimizer import AdamW
from cs336_systems.multihead_flash_attention import MultiheadFlashSelfAttention
from cs336_systems.common import *

from logging import getLogger
logger = getLogger(__name__)

logger.info("initializing...")

warm_up_steps = 3
benchmark_steps = 5


def benchmark(d_model, context_length, batch_szie=8):

    forward_time_sum, backward_time_sum, optimizer_time_sum = 0, 0, 0
    init_mem_sum, peak_mem_sum = 0, 0

    model = MultiheadSelfAttention(
        d_model=d_model,
        num_heads=1,
        max_seq_len=context_length,
        device=DEVICE
    )
    
    optimizer = AdamW(model.parameters())
    
    model.train()
    model.to(DEVICE)

    logger.info("Starting benchmark...")

    for i in range(warm_up_steps + benchmark_steps):

        optimizer.zero_grad()

        if i == warm_up_steps:
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        logger.info(f"Starting {'warmup' if i < warm_up_steps else 'benchmark'} step {i+1 if i < warm_up_steps else i+1 - warm_up_steps}...")

        inputs = torch.rand((batch_szie, context_length, d_model), device=DEVICE)

        init_mem = torch.cuda.memory_allocated(DEVICE)
        t1 = timeit.default_timer()

        with torch.cuda.nvtx.range(f"forward pass"):
            outputs = model(inputs)
            torch.cuda.synchronize()

        t2 = timeit.default_timer()

        with torch.cuda.nvtx.range(f"loss computation"):
            loss = outputs.mean()
            torch.cuda.synchronize()

        peak_mem = torch.cuda.max_memory_allocated(DEVICE)
        t3 = timeit.default_timer()

        with torch.cuda.nvtx.range(f"backward pass"):
            loss.backward()
            torch.cuda.synchronize()

        t4 = timeit.default_timer()


        with torch.cuda.nvtx.range(f"optimizer step"):
            optimizer.step()
            torch.cuda.synchronize()

        t5 = timeit.default_timer()

        if i < warm_up_steps:
            continue

        logger.info(f"  Forward pass: {t2 - t1:.6f} seconds")
        logger.info(f"  Backward pass: {t4 - t3:.6f} seconds")
        logger.info(f"  Optimizer step: {t5 - t4:.6f} seconds")

        forward_time_sum += t2 - t1
        backward_time_sum += t4 - t3
        optimizer_time_sum += t5 - t4
        init_mem_sum += init_mem
        peak_mem_sum += peak_mem

        if i == warm_up_steps + benchmark_steps - 1:
            avg_forward_time = forward_time_sum / benchmark_steps
            avg_backward_time = backward_time_sum / benchmark_steps
            avg_init_mem = init_mem_sum / benchmark_steps
            avg_peak_mem = peak_mem_sum / benchmark_steps
            logger.info(f"Average times over {benchmark_steps} steps:")
            logger.info(f"  Forward pass: {avg_forward_time:.6f} seconds")
            logger.info(f"  Backward pass: {avg_backward_time:.6f} seconds")
            logger.info(f"  Optimizer step: {optimizer_time_sum / benchmark_steps:.6f} seconds")
            logger.info(f"Average memory over {benchmark_steps} steps:")
            logger.info(f"  Initial memory: {avg_init_mem / (1024 * 1024):.2f} MB")
            logger.info(f"  Peak memory: {avg_peak_mem / (1024 * 1024):.2f} MB")
            break

def cal_mem(d_model, context_length, batch_size=8):
    proj = 3 * d_model * context_length * batch_size
    mask = context_length * context_length * batch_size
    attn_scores = context_length * context_length * batch_size * 2
    attn_out = d_model * context_length * batch_size
    out_proj = d_model * context_length * batch_size
    total_mem = (proj + mask + attn_scores + attn_out + out_proj) * 4
    return total_mem / 1024**2 

benchmark(d_model=64, context_length=4096)
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)