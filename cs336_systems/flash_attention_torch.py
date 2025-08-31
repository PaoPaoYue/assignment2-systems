from dataclasses import dataclass
import math

from einops import einsum
import torch

class FlashAttnTorchFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, is_causal:bool=True) -> torch.Tensor:
        assert Q.dim() == K.dim() == V.dim() == 3, "Q, K, V must be 3-dimensional tensors."
        
        ctx.is_causal = is_causal

        B, T, D = Q.size()
        O = torch.empty((B, T, D), device=Q.device, dtype=Q.dtype) # output
        L = torch.empty((B, T), device=Q.device, dtype=Q.dtype) # logsumexp of attention scores

        B_q = FlashAttnTorchFunc._auto_batch_size(T) # query tile size
        B_k = FlashAttnTorchFunc._auto_batch_size(K.size(1)) # key & value tile size
        T_q = math.ceil(T / B_q) # number of query tiles

        args = [FlashAttnTorchFunc._ForwardKernalArgs(b, i, Q, K, V, O, L, B_q, B_k, is_causal) for b in range(B) for i in range(T_q)]

        for arg in args:
            FlashAttnTorchFunc._forwardKernal(arg)
        ctx.save_for_backward(Q, K, V, O, L)

        return O
    

    @dataclass
    class _ForwardKernalArgs:
        b: int # batch index
        i: int # query tile index
        Q: torch.Tensor # [B T D]
        K: torch.Tensor # [B T D]
        V: torch.Tensor # [B T D]
        O: torch.Tensor # output [B T D]
        L: torch.Tensor # logsumexp of attention scores [B T]
        B_q: int # query tile size
        B_k: int # key & value tile size
        is_causal: bool # whether to use causal masking

    @staticmethod
    def _forwardKernal(args: _ForwardKernalArgs):
        T = args.Q.size(1)
        D = args.Q.size(2)

        q_start, q_end = FlashAttnTorchFunc._compute_boundary(args.i, args.B_q, T)
        Q_tile = args.Q[args.b, q_start:q_end, :] # [B_q D]

        m_i = torch.full((2, q_end - q_start), float("-inf"), device=Q_tile.device) # [B_q]
        l_i = torch.zeros((q_end - q_start,), device=Q_tile.device) # [B_q]
        o_i = torch.zeros((q_end - q_start, D), device=Q_tile.device) # [B_q D]

        m_old, m_new = 0, 1

        for j in range(math.ceil(args.K.size(1) / args.B_k)):
            k_start, k_end = FlashAttnTorchFunc._compute_boundary(j, args.B_k, T)
            
            if args.is_causal and k_start >= q_end:
                continue

            K_tile = args.K[args.b, k_start:k_end, :] # [B_k D]
            V_tile = args.V[args.b, k_start:k_end, :] # [B_k D]
            S = einsum(Q_tile, K_tile, "b_q d, b_k d -> b_q b_k") / math.sqrt(D) # [B_q B_k]
            if args.is_causal and k_end > q_start:
                q_idx = torch.arange(q_start, q_end, device=S.device)[:, None]
                k_idx = torch.arange(k_start, k_end, device=S.device)[None, :]
                S = S.masked_fill(q_idx < k_idx, float("-inf"))

            m_i[m_new] = torch.maximum(m_i[m_old], S.max(dim=-1).values) # [B_q]
            P_i = torch.exp(S - m_i[m_new].unsqueeze(-1)) # [B_q B_k]
            delta = torch.exp(m_i[m_old] - m_i[m_new]) # [B_q]
            l_i = delta * l_i + torch.sum(P_i, dim=-1) # [B_q]
            o_i = delta.unsqueeze(-1) * o_i + einsum(P_i, V_tile, "b_q b_k, b_k d -> b_q d") # [B_q D]
            m_old, m_new = m_new, m_old

        args.O[args.b, q_start:q_end, :] = o_i / l_i.unsqueeze(-1) # [B_q D]
        args.L[args.b, q_start:q_end] = torch.log(l_i) + m_i[m_old]

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        B, T, D = Q.size()
        O_dO = (O * dO).sum(dim=-1, keepdim=False) # [B T] 

        dQ = torch.zeros_like(Q) # [B T D]
        dK = torch.zeros_like(K) # [B T D]
        dV = torch.zeros_like(V) # [B T D]

        B_q = FlashAttnTorchFunc._auto_batch_size(T) # query tile size
        B_k = FlashAttnTorchFunc._auto_batch_size(K.size(1)) # key & value tile size
        T_q = math.ceil(T / B_q) # number of query tiles
        T_k = math.ceil(K.size(1) / B_k) # number of key & value tiles

        for b in range(B):
            for i in range(T_q):
                args = FlashAttnTorchFunc._BackwardKernalArgs(b, i, 0, Q, K, V, O, L, dO, O_dO, dQ, dK, dV, B_q, B_k, ctx.is_causal)
                FlashAttnTorchFunc._backwardFirstKernal(args)
        
        for b in range(B):
            for j in range(T_k):
                args = FlashAttnTorchFunc._BackwardKernalArgs(b, 0, j, Q, K, V, O, L, dO, O_dO, dQ, dK, dV, B_q, B_k, ctx.is_causal)
                FlashAttnTorchFunc._backwardSecondKernal(args)

        return dQ, dK, dV, None

    @dataclass
    class _BackwardKernalArgs:
        b: int # batch index
        i: int # query tile index
        j: int # key tile index
        Q: torch.Tensor # [B T D]
        K: torch.Tensor # [B T D]
        V: torch.Tensor # [B T D]
        O: torch.Tensor # output [B T D]
        L: torch.Tensor # logsumexp of attention scores [B T]
        dO: torch.Tensor # gradient of output [B T D]
        O_dO: torch.Tensor # [B T]
        dQ: torch.Tensor # gradient of Q [B T D]
        dK: torch.Tensor # gradient of K [B T D]
        dV: torch.Tensor # gradient of V [B T D]
        B_q: int # query tile size
        B_k: int # key & value tile size
        is_causal: bool # whether to use causal masking

    @staticmethod
    def _backwardFirstKernal(args: _BackwardKernalArgs):
        T = args.Q.size(1)
        D = args.Q.size(2)

        q_start, q_end = FlashAttnTorchFunc._compute_boundary(args.i, args.B_q, T)
        Q_tile = args.Q[args.b, q_start:q_end, :] # [B_q D]
        dO_tile = args.dO[args.b, q_start:q_end, :] # [B_q D]
        L_tile = args.L[args.b, q_start:q_end] # [B_q]
        O_dO_tile = args.O_dO[args.b, q_start:q_end] # [B_q]

        for j in range(math.ceil(args.K.size(1) / args.B_k)):
            k_start, k_end = FlashAttnTorchFunc._compute_boundary(j, args.B_k, T)
            
            if args.is_causal and k_start >= q_end:
                continue

            K_tile = args.K[args.b, k_start:k_end, :] # [B_k D]
            V_tile = args.V[args.b, k_start:k_end, :] # [B_k D]
            S = einsum(Q_tile, K_tile, "b_q d, b_k d -> b_q b_k") / math.sqrt(D) # [B_q B_k]
            if args.is_causal and k_end > q_start:
                q_idx = torch.arange(q_start, q_end, device=S.device)[:, None]
                k_idx = torch.arange(k_start, k_end, device=S.device)[None, :]
                S = S.masked_fill(q_idx < k_idx, float("-inf"))

            P_i = torch.exp(S - L_tile.unsqueeze(-1)) # [B_q B_k]
            dP_i = einsum(dO_tile, V_tile, "b_q d, b_k d -> b_q b_k") # [B_q B_k]
            dS_i = P_i * (dP_i - O_dO_tile.unsqueeze(-1)) / math.sqrt(D) # [B_q B_k]

            args.dQ[args.b, q_start:q_end, :] += einsum(dS_i, K_tile, "b_q b_k, b_k d -> b_q d") # [B_q D]

    @staticmethod
    def _backwardSecondKernal(args: _BackwardKernalArgs):
        T = args.K.size(1)
        D = args.K.size(2)

        k_start, k_end = FlashAttnTorchFunc._compute_boundary(args.j, args.B_k, T)
        K_tile = args.K[args.b, k_start:k_end, :] # [B_k D]
        V_tile = args.V[args.b, k_start:k_end, :] # [B_k D]

        for i in range(math.ceil(T / args.B_q)):
            q_start, q_end = FlashAttnTorchFunc._compute_boundary(i, args.B_k, T)
            
            if args.is_causal and k_start >= q_end:
                continue

            Q_tile = args.Q[args.b, q_start:q_end, :] # [B_q D]
            dO_tile = args.dO[args.b, q_start:q_end, :] # [B_q D]
            L_tile = args.L[args.b, q_start:q_end] # [B_q]
            O_dO_tile = args.O_dO[args.b, q_start:q_end] # [B_q]

            S = einsum(Q_tile, K_tile, "b_q d, b_k d -> b_q b_k") / math.sqrt(D) # [B_q B_k]
            if args.is_causal and k_end > q_start:
                q_idx = torch.arange(q_start, q_end, device=S.device)[:, None]
                k_idx = torch.arange(k_start, k_end, device=S.device)[None, :]
                S = S.masked_fill(q_idx < k_idx, float("-inf"))

            P_i = torch.exp(S - L_tile.unsqueeze(-1)) # [B_q B_k]
            dP_i = einsum(dO_tile, V_tile, "b_q d, b_k d -> b_q b_k") # [B_q B_k]
            dS_i = P_i * (dP_i - O_dO_tile.unsqueeze(-1)) / math.sqrt(D) # [B_q B_k]

            args.dV[args.b, k_start:k_end, :] += einsum(P_i, dO_tile, "b_q b_k, b_q d -> b_k d") # [B_k D]
            args.dK[args.b, k_start:k_end, :] += einsum(dS_i, Q_tile, "b_q b_k, b_q d -> b_k d") # [B_k D]

    @staticmethod
    def _compute_boundary(index:int, tile_size:int, total_size:int) -> tuple[int, int]:
        start = index * tile_size
        end = min((index + 1) * tile_size, total_size)
        return start, end
    
    @staticmethod
    def _auto_batch_size(total_size:int, max_tile_size:int=128, min_tile_size:int=32, expected_max_tiles:int=16) -> int:
        if total_size <= min_tile_size:
            return total_size
        for tile_size in range(min_tile_size, max_tile_size + 1, 32):
            if math.ceil(total_size / tile_size) <= expected_max_tiles:
                return tile_size
        return min_tile_size

    # no tiling backward    
    # @staticmethod
    # def backward(ctx, d_O):
    #     Q, K, V, O, L = ctx.saved_tensors
    #     B, T, D = Q.size()
    #     O_dO = (O * d_O).sum(dim=-1, keepdim=False) # [B T] 
    #     S = einsum(Q, K, '... q d, ... k d -> ... q k') / math.sqrt(D) # [B T T] 
    #     P = torch.exp(S - L.unsqueeze(-1)) # [B T T]
    #     d_V = einsum(P, d_O, '... q k, ... q d -> ... k d') # [B T D] 
    #     d_P = einsum(d_O, V, '... q d, ... k d -> ... q k') # [B T T] 
    #     d_S = P * (d_P - O_dO.unsqueeze(-1)) # [B T T] 
    #     if ctx.is_causal:
    #         mask = torch.ones((T, T), device=d_S.device)
    #         mask = torch.tril(mask)
    #         d_S = d_S.masked_fill(mask == 0, 0.0)
    #     d_Q = einsum(d_S, K, '... q k, ... k d -> ... q d') / math.sqrt(D) # [B T D] 
    #     d_K = einsum(d_S, Q, '... q k, ... q d -> ... k d') / math.sqrt(D) # [B T D]
    #     return d_Q, d_K, d_V, None
