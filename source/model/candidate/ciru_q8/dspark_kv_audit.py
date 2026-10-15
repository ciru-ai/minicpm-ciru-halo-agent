"""Sample actual draft context writes against per-layer native BF16 preparation."""
import json
import torch
import torch.nn.functional as F
from vllm.v1.attention.ops.paged_attn import PagedAttention

def install(worker,proposer):
    original=proposer.propose
    count=[0]
    def propose(input_batch,*args,**kwargs):
        result=original(input_batch,*args,**kwargs)
        if kwargs.get('dummy_run',False) or getattr(worker,'_dspark_warmup',False):return result
        if count[0]>=20:return result
        count[0]+=1
        inner=proposer.model.model
        n=input_batch.num_tokens
        slots=proposer._context_slot_mappings[0,:n]
        valid=torch.where(slots>=0)[0]
        if not valid.numel():return result
        ix=valid[torch.linspace(0,valid.numel()-1,min(16,valid.numel()),device=valid.device).long()]
        x=proposer.hidden_states[ix]
        positions=proposer.context_positions[ix]
        slots=slots[ix]
        def norm(x,w,eps):
            z=x.float();return ((z*torch.rsqrt(z.square().mean(-1,keepdim=True)+eps)).to(x.dtype)*w).to(x.dtype)
        z=norm(x,inner.hidden_norm.weight,inner.hidden_norm.variance_epsilon)
        records=[]
        for i,layer in enumerate(inner.layers):
            attn=layer.self_attn
            kv=F.linear(z,attn.qkv_proj.weight[attn.q_size:],None)
            k,v=kv.split(attn.kv_size,-1)
            k=norm(k.reshape(-1,attn.num_kv_heads,attn.head_dim),attn.k_norm.weight,attn.k_norm.variance_epsilon).reshape(-1,attn.kv_size)
            k,_=attn.rotary_emb.forward_native(positions,k,None)
            k=k.reshape(-1,attn.num_kv_heads,attn.head_dim)
            v=v.reshape_as(k)
            kc,vc=PagedAttention.split_kv_cache(attn.attn.kv_cache.transpose(0,1),attn.num_kv_heads,attn.head_dim)
            blocks=slots//kc.shape[3];offset=slots%kc.shape[3]
            actual_k=kc[blocks,:,:,offset,:].reshape_as(k)
            actual_v=vc[blocks,:,:,offset].reshape_as(v)
            def stats(a,b):
                delta=a.float()-b.float()
                return dict(max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()),nonfinite=int((~torch.isfinite(a)).sum()),reference_rms=float(b.float().square().mean().sqrt()))
            records.append(dict(layer=i,k=stats(actual_k,k),v=stats(actual_v,v),rope_dim=attn.rotary_emb.rotary_dim))
        row=dict(step=count[0],num_target_tokens=n,num_valid_context=int(valid.numel()),positions=positions.cpu().tolist(),slots=slots.cpu().tolist(),layers=records)
        with (worker.run/'draft-kv-preparation-audit.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return result
    proposer.propose=propose
