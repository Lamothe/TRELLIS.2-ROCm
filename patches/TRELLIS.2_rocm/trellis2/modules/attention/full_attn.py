import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    # (B, L, H, D) -> (B, H, L, D)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    # (B, H, L, D) -> (B, L, H, D) -> (B, L, H*D)
    out = out.transpose(1, 2).contiguous()
    return out.reshape(out.shape[0], out.shape[1], -1)
