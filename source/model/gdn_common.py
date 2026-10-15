"""Actual-operand serialization; preserve strides and offsets, not unused storage."""
import hashlib, inspect, math, pathlib, torch

def clone_layout(x, device):
    if isinstance(x, torch.Tensor):
        assert all(s >= 0 for s in x.stride())
        size = x.storage_offset() + 1 + sum((n-1)*s for n,s in zip(x.shape,x.stride())) if x.numel() else x.storage_offset()
        base = torch.empty(size, dtype=x.dtype, device=device)
        result = base.as_strided(x.shape, x.stride(), x.storage_offset())
        result.copy_(x)
        return result
    if isinstance(x, dict): return {k:clone_layout(v,device) for k,v in x.items()}
    if isinstance(x, tuple): return tuple(clone_layout(v,device) for v in x)
    if isinstance(x, list): return [clone_layout(v,device) for v in x]
    return x

def describe(x):
    if isinstance(x, torch.Tensor):
        return dict(shape=list(x.shape),stride=list(x.stride()),offset=x.storage_offset(),dtype=str(x.dtype),device=str(x.device))
    if isinstance(x,dict): return {k:describe(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [describe(v) for v in x]
    return str(x) if isinstance(x,torch.dtype) else x

def bindings():
    from vllm.third_party.flash_linear_attention.ops import chunk
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen
    core=qwen.QwenGatedDeltaNetAttention._forward_core.__globals__
    stages={s:(chunk.chunk_gated_delta_rule_fwd.__globals__,s) for s in ('recompute_w_u_fwd','chunk_fwd_o')}
    stages['postconv']=(core,'fused_post_conv_prep')
    return stages

def identity(fn):
    p=pathlib.Path(inspect.getfile(inspect.unwrap(fn)))
    return dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),signature=str(inspect.signature(fn)))
