import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

def broadcast_model_state(model, src=0):
    """
    将 rank=src 的模型 state_dict 广播到所有进程
    包括参数和缓冲区，确保模型完全一致。

    Args:
        model (torch.nn.Module): 需要同步的模型
        src (int): 源 rank（通常是 0）
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized.")

    # 遍历 state_dict 中的所有张量（参数 + 缓冲区）
    for tensor in model.state_dict().values():
        # 确保张量在 GPU 上（如果是 CUDA 训练）
        if torch.is_tensor(tensor):
            dist.broadcast(tensor.data, src=src)

class NaiveDDPWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
        broadcast_model_state(self.module)

    def forward(self, inputs, **kwargs):
        return self.module(inputs, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        handles = []
        for param in self.module.parameters():
            if param.grad is not None:
                handles.append(dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True))
                param.grad /= dist.get_world_size()

        for handle in handles:
            handle.wait()
        torch.cuda.synchronize()

class MinimalDDPFlatWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
        broadcast_model_state(self.module)

    def forward(self, inputs, **kwargs):
        return self.module(inputs, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        grads = [p.grad for p in self.module.parameters() if p.grad is not None]
        if not grads:
            return
        flat_grads = _flatten_dense_tensors(grads)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= dist.get_world_size()
        synced_grads = _unflatten_dense_tensors(flat_grads, grads)
        for g, synced in zip(grads, synced_grads):
            g.copy_(synced)
        torch.cuda.synchronize()

class DDPOverlapWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
        broadcast_model_state(self.module)
        self.handles = []
        for param in model.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._sync_individual_parameters_grad)

    def forward(self, inputs, **kwargs):
        return self.module(inputs, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        for param, handle in self.handles:
            handle.wait()
            param.grad /= dist.get_world_size()

        self.handles.clear()

        torch.cuda.synchronize()

    @torch.no_grad()
    def _sync_individual_parameters_grad(self, p: torch.Tensor):
        self.handles.append((p, dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)))

class DDPOverlapBucketedWrapper(torch.nn.Module):
    def __init__(self, model, bucket_size_mb: float):
        super().__init__()
        self.module = model
        broadcast_model_state(self.module)
        self.bucket_bytes = int(bucket_size_mb * 1024 * 1024)
        self.current_bucket = []
        self.current_bucket_size = 0
        self.handles = []
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        for p in model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._grad_hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        self._flush_bucket()

        for flat_bucket, origin_bucket, handle in self.handles:
            if handle is not None:
                handle.wait()
                flat_bucket /= self.world_size
            offset = 0
            for grad in origin_bucket:
                n = grad.numel()
                grad.copy_(flat_bucket[offset:offset+n].view_as(grad))
                offset += n

        self.handles.clear()
        self.current_bucket.clear()
        self.current_bucket_size = 0

    @torch.no_grad()
    def _grad_hook(self, p: torch.Tensor):
        g = p.grad
        if g is None:
            return
        for chunk in self._split_into_chunks(g):
            sz = chunk.numel() * chunk.element_size()
            if self.current_bucket_size + sz > self.bucket_bytes:
                self._flush_bucket()
            self.current_bucket.append(chunk)
            self.current_bucket_size += sz

    def _split_into_chunks(self, t: torch.Tensor):
        max_elems = max(1, self.bucket_bytes // t.element_size())
        flat = t.contiguous().view(-1)
        return [flat[i:i+max_elems] for i in range(0, flat.numel(), max_elems)]

    def _flush_bucket(self):
        if not self.current_bucket:
            return
        flat_bucket = torch._utils._flatten_dense_tensors(
            [t for t in self.current_bucket]
        )
        handle = None
        if self.world_size > 1 and dist.is_initialized():
            handle = dist.all_reduce(flat_bucket, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((flat_bucket, list(self.current_bucket), handle))
        self.current_bucket.clear()
        self.current_bucket_size = 0
