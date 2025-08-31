import math
from einops import rearrange
import torch
import triton
import triton.language as tl

def get_autotune_config():
    # combination of [128 64 32]
    return [
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}),
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}),
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 32}),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 128}),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 128}),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 64}),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['N_QUERIES', 'N_KEYS'],
)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):

    i = tl.program_id(0)
    b = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + b * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_T_block_ptr = tl.make_block_ptr(
        K_ptr + b * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + b * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + b * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + b * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(i * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)

    m_old = tl.full((Q_TILE_SIZE,), float("-inf"), tl.float32)
    m_new = tl.full((Q_TILE_SIZE,), float("-inf"), tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)): 
        q_start = i * Q_TILE_SIZE
        q_end   = (i + 1) * Q_TILE_SIZE
        k_start = j * K_TILE_SIZE
        k_end   = (j + 1) * K_TILE_SIZE

        if is_causal and k_start >= q_end:
            pass
        else:
            K_T_tile = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
            V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            S = tl.dot(Q_tile, K_T_tile) * scale
            if is_causal and k_end > q_start:
                q_idx = q_start + tl.arange(0, Q_TILE_SIZE)[:, None]
                k_idx = k_start + tl.arange(0, K_TILE_SIZE)[None, :]
                S = tl.where(q_idx < k_idx, float("-inf"), S)

            m_new = tl.maximum(m_old, tl.max(S, axis=1))
            p_i = tl.exp(S - m_new[:, None])
            delta = tl.exp(m_old - m_new)
            l_i = delta * l_i + tl.sum(p_i, axis=1)

            o_i = delta[:, None] * o_i + tl.dot(p_i, V_tile)
            m_old = m_new

            K_T_block_ptr = K_T_block_ptr.advance((0, K_TILE_SIZE))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(O_block_ptr, o_i / l_i[:, None], boundary_check=(0, 1))
    tl.store(L_block_ptr, tl.log(l_i) + m_old, boundary_check=(0,))


class FlashAttnTritonFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, is_causal:bool=True) -> torch.Tensor:
        """
        Q [B T D]
        K [B T D]
        V [B T D]
        """
        assert Q.dim() == K.dim() == V.dim() == 3, "Q, K, V must be 3-dimensional tensors."
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Q, K, V must be CUDA tensors."
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Q, K, V must be contiguous tensors."

        ctx.is_causal = is_causal

        B, T, D = Q.size()
        O = torch.empty((B, T, D), device=Q.device, dtype=Q.dtype) # output
        L = torch.empty((B, T), device=Q.device, dtype=Q.dtype) # logsumexp of attention scores

        grid = lambda META: (triton.cdiv(T, META['Q_TILE_SIZE']), B)

        flash_fwd_kernel[grid](
            Q, K, V, 
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Q.size(1), K.size(1),
            math.pow(D, -0.5),
            D,
            is_causal=is_causal
        )
        torch.cuda.synchronize()
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    
    @staticmethod
    def _auto_batch_size(total_size:int, max_tile_size:int=128, min_tile_size:int=32, expected_max_tiles:int=16) -> int:
        if total_size <= min_tile_size:
            return total_size
        for tile_size in range(min_tile_size, max_tile_size + 1, 32):
            if math.ceil(total_size / tile_size) <= expected_max_tiles:
                return tile_size
        return min_tile_size

    @staticmethod
    def backward(ctx, d_O):
        raise NotImplementedError("Backward pass is not implemented yet.")
    