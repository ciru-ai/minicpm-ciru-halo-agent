"""Draft-only noncausal H8 partition, paired against stock on identical live operands."""
import hashlib,json,pathlib,os
import torch
from .dspark_attention_partition import forward as partition

def install(worker,mode):
    assert mode in ('stock','audit','retained')
    window=int(os.environ.get('CIRU_DSPARK_WINDOW','0'))
    assert window==0 or window>=1024
    assert not window or mode!='stock', 'Window diagnostic requires matched partition/reference path'
    from vllm.v1.attention.backends import rocm_attn
    parent=rocm_attn.chunked_prefill_paged_decode
    worker._draft_attention_stats=torch.zeros(5,dtype=torch.float32,device=worker.device) if mode=='audit' else None
    worker._draft_attention_seen=set()
    if mode=='stock':return
    def forward(query,key,value,output,kv_cache_dtype,key_cache,value_cache,block_table,
                query_start_loc,seq_lens,max_seq_len,max_query_len,k_scale,v_scale,
                alibi_slopes=None,sliding_window=None,sm_scale=None,output_scale=None,
                sinks=None,is_block_table_ptr=False,causal=True):
        page=key_cache.shape[3] if key_cache.ndim==5 else 0
        eligible=(not causal and query.shape[1:]==(8,256) and query.shape[0]<=128
            and page in (1056,1072,1088,1120,1152,1168,1184,2240) and key_cache.shape[1:]==(2,32,page,8)
            and value_cache.shape[1:]==(2,256,page) and 1<=seq_lens.numel()<=8 and 1<=max_query_len<=16
            and query.dtype==key_cache.dtype==value_cache.dtype==output.dtype==torch.bfloat16
            and kv_cache_dtype=='auto' and not is_block_table_ptr and alibi_slopes is None
            and sinks is None and output_scale is None and (sliding_window is None or sliding_window<=0))
        def original(out):
            return parent(query,key,value,out,kv_cache_dtype,key_cache,value_cache,block_table,
                query_start_loc,seq_lens,max_seq_len,max_query_len,k_scale,v_scale,
                alibi_slopes,sliding_window,sm_scale,output_scale,sinks,is_block_table_ptr,causal)
        if not eligible:return original(output)
        if window:
            # Drop complete old pages for the drafter only. Both audit paths
            # consume this same trimmed table; positions/RoPE stay absolute.
            shift=(seq_lens-window).clamp_min(0)//page
            columns=torch.arange(block_table.shape[1],device=block_table.device)
            source=columns[None,:]+shift[:,None]
            block_table=block_table.gather(1,source.clamp_max(block_table.shape[1]-1).long())
            seq_lens=seq_lens-shift*page
            max_seq_len=min(max_seq_len,window+page-1)
        partition(query,key_cache,value_cache,output,block_table,query_start_loc,seq_lens,
                  sm_scale if sm_scale is not None else .0625,k_scale,v_scale,max_query_len=max_query_len)
        stats=worker._draft_attention_stats
        if stats is not None:
            reference=torch.empty_like(output);original(reference)
            live=torch.arange(query.shape[0],device=query.device)<query_start_loc[-1];active=live[:,None,None]
            diff=torch.where(active,output.float()-reference.float(),0.)
            stats[0].add_(live.any());stats[1].copy_(torch.maximum(stats[1],diff.abs().nan_to_num().amax()))
            rms=(diff.square().sum()/(live.sum()*2048).clamp_min(1)).sqrt()
            stats[2].copy_(torch.maximum(stats[2],rms.nan_to_num()))
            stats[3].add_((active&~torch.isfinite(output)).sum()+(active&~torch.isfinite(reference)).sum())
            stats[4].add_(live.sum()*2048)
        sig=(query.shape[0],max_query_len,seq_lens.numel(),page,torch.cuda.is_current_stream_capturing())
        if sig not in worker._draft_attention_seen:
            worker._draft_attention_seen.add(sig)
            with (worker.run/'draft-attention-calls.jsonl').open('a') as f:f.write(json.dumps(dict(signature=sig,causal=causal,mode=mode,scope='Host call/capture; actual graph replay included in audit operand statistics'))+'\n')
        return output
    rocm_attn.chunked_prefill_paged_decode=forward
    (worker.run/'draft-attention-binding.json').write_text(json.dumps(dict(mode=mode,draft_window_min_tokens=window,draft_window_granularity='complete pages; actual window < minimum + page size',source_sha256=hashlib.sha256(pathlib.Path(__file__).with_name('dspark_attention_partition.py').read_bytes()).hexdigest(),delta='H8/KV2 instead of H16/KV2; every query uses full request sequence length; target operator unchanged',qualified=False),indent=2)+'\n')
