from einops import rearrange
import torch

from cs336_basics.nn_model import Linear, Rope
from cs336_systems.flash_attention_triton import FlashAttnTritonFunc

class MultiheadFlashSelfAttention(torch.nn.Module):
    def __init__(self, 
        d_model: int,
        num_heads: int,
        max_seq_len: int = 512,
        theta: float = 0,
        device=None, 
        dtype=None
    ): 
        super(MultiheadFlashSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope =  Rope(theta, d_model // num_heads, max_seq_len, device=device) if theta > 0 else None
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None, is_causal: bool = True) -> torch.Tensor:
        q = rearrange(self.q_proj(x), "... seq_len (num_heads d_k) -> (... num_heads) seq_len d_k", num_heads=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq_len (num_heads d_k) -> (... num_heads) seq_len d_k", num_heads=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq_len (num_heads d_k) -> (... num_heads) seq_len d_k", num_heads=self.num_heads)
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)        

        attn_output = FlashAttnTritonFunc.apply(q, k, v, is_causal)
        attn_output = rearrange(attn_output, "(batch num_heads) seq_len d_k -> batch seq_len (num_heads d_k)", num_heads=self.num_heads)

        return self.output_proj(attn_output)
