"""Retained partitioned decode on the pinned engine's native cache pages."""
import hashlib,json,pathlib
import torch
from .attention_partition import forward as partition

def install(worker,mode):
    from vllm.v1.attention.backends import rocm_attn
    parent=rocm_attn.chunked_prefill_paged_decode
    stats=torch.zeros(5,dtype=torch.float32,device=worker.device) if mode=='audit' else None
    worker._attention_stats=stats;seen=set();worker._attention_dispatch_seen=seen
    def forward(query,key,value,output,kv_cache_dtype,key_cache,value_cache,block_table,
                query_start_loc,seq_lens,max_seq_len,max_query_len,k_scale,v_scale,
                alibi_slopes=None,sliding_window=None,sm_scale=None,output_scale=None,
                sinks=None,is_block_table_ptr=False,causal=True):
        page=key_cache.shape[3] if key_cache.ndim==5 else 0
        eligible=(page in (1056,1072,1088,1120,1152,1168,1184,2240) and key_cache.shape[1:]==(2,32,page,8)
          and value_cache.shape[1:]==(2,256,page) and query.shape[1:]==(16,256)
          and query.shape[0]<=128 and 1<=seq_lens.numel()<=8 and 1<=max_query_len<=16
          and query.dtype==key_cache.dtype==value_cache.dtype==output.dtype==torch.bfloat16
          and kv_cache_dtype=='auto' and causal and not is_block_table_ptr and alibi_slopes is None
          and sinks is None and output_scale is None and (sliding_window is None or sliding_window<=0))
        def original(out):
            return parent(query,key,value,out,kv_cache_dtype,key_cache,value_cache,block_table,
                query_start_loc,seq_lens,max_seq_len,max_query_len,k_scale,v_scale,
                alibi_slopes,sliding_window,sm_scale,output_scale,sinks,is_block_table_ptr,causal)
        if not eligible:return original(output)
        partition(query,key_cache,value_cache,output,block_table,query_start_loc,seq_lens,
            sm_scale if sm_scale is not None else .0625,k_scale,v_scale,max_query_len=max_query_len)
        if stats is not None:
            reference=torch.empty_like(output);original(reference)
            live=torch.arange(query.shape[0],device=query.device)<query_start_loc[-1]
            active=live[:,None,None]
            diff=torch.where(active,output.float()-reference.float(),0.)
            stats[0].add_(live.any())
            stats[1].copy_(torch.maximum(stats[1],diff.abs().nan_to_num().amax()))
            rms=(diff.square().sum()/(live.sum()*4096).clamp_min(1)).sqrt()
            stats[2].copy_(torch.maximum(stats[2],rms.nan_to_num()))
            stats[3].add_((active&~torch.isfinite(output)).sum()+(active&~torch.isfinite(reference)).sum())
            stats[4].add_(live.sum()*4096)
        signature=(query.shape[0],max_query_len,seq_lens.numel(),page,torch.cuda.is_current_stream_capturing())
        if signature not in seen:
            seen.add(signature)
            with (worker.run/'attention-calls.jsonl').open('a') as f:f.write(json.dumps(dict(rows=signature[0],max_query=max_query_len,requests=signature[2],page=page,capture=signature[4],mode=mode,source='ciru_q8.attention_partition',scope='Host invocation/capture; graph replay statistics collected only in audit mode'))+'\n')
        return output
    rocm_attn.chunked_prefill_paged_decode=forward
    (worker.run/'attention-binding.json').write_text(json.dumps(dict(mode=mode,source_sha256=hashlib.sha256(pathlib.Path(__file__).with_name('attention_partition.py').read_bytes()).hexdigest(),weights_changed=False,kv_layout_changed=False,bulk_prefill_unchanged=True,eligibility='Matching target or own-MTP BF16 attention geometry with max_query_len<=16, including short cached-prefill tails',quality_qualified=False,full_optimizer_qualified=False),indent=2)+'\n')
