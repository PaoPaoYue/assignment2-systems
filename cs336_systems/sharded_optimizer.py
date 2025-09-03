import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Any, Dict, Iterable, List, Tuple, Type, cast
import math

class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # 记录 (原param, shard_size)
        self._shard_meta: List[Tuple[torch.nn.Parameter, int]] = []
        self._local_params: List[torch.nn.Parameter] = []

        super().__init__(params, {})  # 初始化父类

        # 构造内部优化器
        self._inner_optim = optimizer_cls(self._local_params, **kwargs)

    def get_inner_optimizer(self) -> Optimizer:
        return self._inner_optim

    @torch.no_grad()
    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        # 1. 类型检查
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, got {type(param_group)}")

        # 2. 必须包含 'params' 键
        if 'params' not in param_group:
            raise ValueError("param_group must contain a 'params' key")

        raw_params = param_group['params']

        # 3. 统一成 list
        if isinstance(raw_params, torch.Tensor):
            raw_params = [raw_params]
        elif isinstance(raw_params, Iterable):
            raw_params = list(raw_params)
        else:
            raise TypeError("'params' must be a Parameter or an iterable of Parameters")

        local_group_params = []

        for p in raw_params:
            # 4. 必须是 Parameter
            if not isinstance(p, torch.Tensor):
                raise TypeError(f"Expected torch.nn.Parameter, got {type(p)}")

            # 5. 跳过不需要梯度的参数
            if not p.requires_grad:
                continue

            # 6. 不能是标量（0-D tensor）
            if p.ndim == 0:
                raise ValueError(f"Cannot shard scalar parameter with shape {p.shape}")

            # 7. 分片逻辑
            shard_size = math.ceil(p.shape[0] / self.world_size)
            start = self.rank * shard_size

            shard_param = torch.nn.Parameter(p[start:start+shard_size])
            local_group_params.append(shard_param)

            # 记录元信息
            self._shard_meta.append((p, shard_size))
            self._local_params.append(shard_param)

        super().add_param_group(param_group)

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        # 记录梯度
        for (origin_param, shard_size), shard_param in zip(self._shard_meta, self._local_params):
            start, end = self.rank * shard_size, (self.rank + 1) * shard_size
            if shard_param.grad is None:
                shard_param.grad = torch.zeros_like(shard_param.data)
            shard_param.grad.data.copy_(origin_param.grad.data[start:end])

        loss = self._inner_optim.step(closure=closure, **kwargs)

        # 同步所有切片
        if self.world_size > 1 and dist.is_initialized():
            for (origin_param, shard_size), shard_param in zip(self._shard_meta, self._local_params):
                # 收集所有 rank 的切片
                local_data = shard_param.data
                padding_length = shard_size - local_data.shape[0]
                if padding_length > 0:
                    padding = torch.zeros((padding_length, *local_data.shape[1:]), device=local_data.device)
                    local_data = torch.cat([local_data, padding], dim=0)
                gathered = [torch.empty_like(local_data) for _ in range(self.world_size)]
                dist.all_gather(gathered, local_data)
                # 拼接并截取有效长度（最后一个 rank 可能不足 shard_size）
                if shard_size * self.world_size > origin_param.shape[0]:
                    full = torch.cat(gathered, dim=0)[:origin_param.shape[0]]
                else:
                    full = torch.cat(gathered, dim=0)
                origin_param.data.copy_(full)
        else:
            pass
        return loss