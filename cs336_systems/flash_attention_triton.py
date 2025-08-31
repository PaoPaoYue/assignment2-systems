import math
from einops import rearrange
import torch
import triton
import triton.language as tl

class FlashAttnTritonFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, is_causal:bool=True) -> torch.Tensor:
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
    def backward(ctx, dO):
        assert dO.is_contiguous(), "dO must be contiguous"

        Q, K, V, O, L = ctx.saved_tensors
        B, T, D = Q.shape
        O_dO = (O * dO).sum(dim=-1, keepdim=False) # [B T] 

        dQ = torch.zeros_like(Q) # [B T D]
        dK = torch.zeros_like(K) # [B T D]
        dV = torch.zeros_like(V) # [B T D]

        grid_i = lambda META: (triton.cdiv(T, META['Q_TILE_SIZE']), B)
        grid_j = lambda META: (triton.cdiv(K.size(1), META['K_TILE_SIZE']), B)

        flash_backward_first_kernel[grid_i](
            Q, K, V, L,
            dO, O_dO, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            Q.size(1), K.size(1),
            math.pow(D, -0.5),
            D,
            is_causal=ctx.is_causal
        )
        flash_backward_second_kernel[grid_j](
            Q, K, V, L,
            dO, O_dO, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            Q.size(1), K.size(1),
            math.pow(D, -0.5),
            D,
            is_causal=ctx.is_causal
        )
        torch.cuda.synchronize()
        return dQ, dK, dV, None    

def get_autotune_config():
    # combination of [128 64 32]
    return [
        # triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}),
        # triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}),
        # triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 32}),
        # triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 128}),
        # triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}),
        # triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}),
        # triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 128}),
        # triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 64}),
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
            else:
                k_idx = k_start + tl.arange(0, K_TILE_SIZE)[None, :]
                S = tl.where(k_idx >= N_KEYS, float("-inf"), S)

            m_new = tl.maximum(m_old, tl.max(S, axis=1))
            P_i = tl.exp(S - m_new[:, None])
            delta = tl.exp(m_old - m_new)
            l_i = delta * l_i + tl.sum(P_i, axis=1)

            o_i = delta[:, None] * o_i + tl.dot(P_i, V_tile)
            m_old = m_new

            K_T_block_ptr = K_T_block_ptr.advance((0, K_TILE_SIZE))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(O_block_ptr, o_i / l_i[:, None], boundary_check=(0, 1))
    tl.store(L_block_ptr, tl.log(l_i) + m_old, boundary_check=(0,))



@triton.autotune(
    configs=get_autotune_config(),
    key=['N_QUERIES', 'N_KEYS'],
)
@triton.jit
def flash_backward_first_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr,
    dO_ptr, O_dO_ptr, dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
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
    L_block_ptr = tl.make_block_ptr(
        L_ptr + b * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(i * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + b * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_dO_block_ptr = tl.make_block_ptr(
        O_dO_ptr + b * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(i * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + b * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)
    dO_tile = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)
    L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")  # (Q_TILE_SIZE,)
    O_dO_tile = tl.load(O_dO_block_ptr, boundary_check=(0,), padding_option="zero")  # (Q_TILE_SIZE,)

    dQ_tile = tl.zeros((Q_TILE_SIZE, D), tl.float32)

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
            else:
                k_idx = k_start + tl.arange(0, K_TILE_SIZE)[None, :]
                S = tl.where(k_idx >= N_KEYS, float("-inf"), S)

            P_i = tl.exp(S - L_tile[:, None])
            dP_i = tl.dot(dO_tile, tl.trans(V_tile))
            dS = P_i * (dP_i - O_dO_tile[:, None]) * scale

            dQ_tile = tl.dot(dS, tl.trans(K_T_tile), acc=dQ_tile)

            K_T_block_ptr = K_T_block_ptr.advance((0, K_TILE_SIZE))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ_tile, boundary_check=(0, 1))


@triton.autotune(
    configs=get_autotune_config(),
    key=['N_QUERIES', 'N_KEYS'],
)
@triton.jit
def flash_backward_second_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr,
    dO_ptr, O_dO_ptr, 
    dK_ptr, dV_ptr, 
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):

    j = tl.program_id(0)
    b = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + b * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_T_block_ptr = tl.make_block_ptr(
        K_ptr + b * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, j * K_TILE_SIZE),
        block_shape=(D, K_TILE_SIZE),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + b * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(j * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + b * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + b * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_dO_block_ptr = tl.make_block_ptr(
        O_dO_ptr + b * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + b * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(j * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + b * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(j * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    K_T_tile = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    dV_tile = tl.zeros((K_TILE_SIZE, D), tl.float32)
    dK_tile = tl.zeros((K_TILE_SIZE, D), tl.float32)

    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)): 

        q_start = i * Q_TILE_SIZE
        q_end   = (i + 1) * Q_TILE_SIZE
        k_start = j * K_TILE_SIZE
        k_end   = (j + 1) * K_TILE_SIZE

        if is_causal and k_start >= q_end:
            pass
        else:
            Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)
            dO_tile = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)
            L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")  # (Q_TILE_SIZE,)
            O_dO_tile = tl.load(O_dO_block_ptr, boundary_check=(0,), padding_option="zero")  # (Q_TILE_SIZE,)

            S = tl.dot(Q_tile, K_T_tile) * scale
            if is_causal and k_end > q_start:
                q_idx = q_start + tl.arange(0, Q_TILE_SIZE)[:, None]
                k_idx = k_start + tl.arange(0, K_TILE_SIZE)[None, :]
                S = tl.where(q_idx < k_idx, float("-inf"), S)
            else:
                k_idx = k_start + tl.arange(0, K_TILE_SIZE)[None, :]
                S = tl.where(k_idx >= N_KEYS, float("-inf"), S)

            P_i = tl.exp(S - L_tile[:, None])
            dP_i = tl.dot(dO_tile, tl.trans(V_tile))
            dS = P_i * (dP_i - O_dO_tile[:, None]) * scale


            dV_tile = tl.dot(tl.trans(P_i), dO_tile, acc=dV_tile)  # [K_TILE_SIZE D]
            dK_tile = tl.dot(tl.trans(dS), Q_tile, acc=dK_tile)  # [K_TILE_SIZE D]

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        O_dO_block_ptr = O_dO_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dV_block_ptr, dV_tile, boundary_check=(0, 1))
    tl.store(dK_block_ptr, dK_tile, boundary_check=(0, 1))
