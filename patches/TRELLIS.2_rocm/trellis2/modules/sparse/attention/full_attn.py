import torch
import torch.nn.functional as F

def sparse_scaled_dot_product_attention(*args, **kwargs):
    attn_mask = kwargs.get('attn_mask', None)
    dropout_p = kwargs.get('dropout_p', 0.0)
    is_causal = kwargs.get('is_causal', False)
    scale = kwargs.get('scale', None)

    def unwrap(x):
        return x.feats if hasattr(x, 'feats') else x

    if len(args) == 1:
        output_template = args[0]
        q, k, v = unwrap(args[0]).unbind(-3)
    elif len(args) == 3:
        output_template = args[0]
        q, k, v = unwrap(args[0]), unwrap(args[1]), unwrap(args[2])
    else:
        raise ValueError(f"Expected 1 or 3 args, got {len(args)}")

    # (..., L, H, D) -> (..., H, L, D)
    q, k, v = q.transpose(-2, -3), k.transpose(-2, -3), v.transpose(-2, -3)
    
    out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale)
    
    # Restore -> Force 2D (TotalVoxels, Channels)
    out = out.transpose(-2, -3).contiguous()
    out = out.reshape(-1, out.shape[-2] * out.shape[-1])
    
    return output_template.replace(out) if hasattr(output_template, 'replace') else out
