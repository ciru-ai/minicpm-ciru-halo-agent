"""Reuse IU4's retained BF16 compact attention on native Q8 cache pages."""
import hashlib,json,pathlib,os
import torch
from . import compact_prefill

def install(worker,mode):
    from vllm.v1.attention.backends import rocm_attn
    original=rocm_attn.chunked_prefill_paged_decode
    os.environ['ORNITH_COMPACT_C1_BACKING']='1'
    compact_prefill.configure(max_num_seqs=worker.vllm_config.scheduler_config.max_num_seqs,
        max_model_len=worker.vllm_config.model_config.max_model_len,
        max_num_batched_tokens=worker.vllm_config.scheduler_config.max_num_batched_tokens,
        device=worker.device,iu4_prefill_library=None)
    worker._prefill_stats=torch.zeros(5,device=worker.device) if mode=='audit' else None
    worker._prefill_real=False
    seen=set()
    def forward(query,key,value,output,kv_cache_dtype,key_cache,value_cache,block_table,
                query_start_loc,seq_lens,max_seq_len,max_query_len,k_scale,v_scale,
                alibi_slopes=None,sliding_window=None,sm_scale=None,output_scale=None,
                sinks=None,is_block_table_ptr=False,causal=True):
        def stock(out):
            return original(query,key,value,out,kv_cache_dtype,key_cache,value_cache,block_table,
                query_start_loc,seq_lens,max_seq_len,max_query_len,k_scale,v_scale,
                alibi_slopes,sliding_window,sm_scale,output_scale,sinks,is_block_table_ptr,causal)
        eligible=(max_query_len>16 and causal and kv_cache_dtype=='auto' and not is_block_table_ptr
            and alibi_slopes is None and sinks is None and output_scale is None
            and (sliding_window is None or sliding_window<=0))
        if not eligible:return stock(output)
        audit=(mode=='audit' and worker._prefill_real and not torch.cuda.is_current_stream_capturing())
        signature=(query.shape[0],seq_lens.numel(),max_query_len,max_seq_len,key_cache.shape[3])
        first=signature not in seen
        events=[torch.cuda.Event(enable_timing=True) for _ in range(3)] if audit else None
        if audit:events[0].record()
        used=compact_prefill.try_forward(query,key_cache,value_cache,output,block_table,query_start_loc,
            seq_lens,max_query_len=max_query_len,max_seq_len=max_seq_len,sm_scale=sm_scale or .0625)
        if not used:return stock(output)
        if first:
            seen.add(signature)
            with (worker.run/'prefill-calls.jsonl').open('a') as f:
                f.write(json.dumps(dict(rows=signature[0],requests=signature[1],max_query=max_query_len,max_seq=max_seq_len,
                    page=signature[4],mode=mode,real=worker._prefill_real,source='ciru_q8.compact_prefill'))+'\n')
        if audit:
            events[1].record();reference=torch.empty_like(output);stock(reference);events[2].record();events[2].synchronize()
            live=torch.arange(query.shape[0],device=query.device)<query_start_loc[-1];active=live[:,None,None]
            diff=torch.where(active,output.float()-reference.float(),0.)
            stats=worker._prefill_stats
            stats[0].add_(live.any());stats[1].copy_(torch.maximum(stats[1],diff.abs().nan_to_num().amax()))
            rms=(diff.square().sum()/(live.sum()*4096).clamp_min(1)).sqrt()
            stats[2].copy_(torch.maximum(stats[2],rms.nan_to_num()))
            stats[3].add_((active&~torch.isfinite(output)).sum()+(active&~torch.isfinite(reference)).sum())
            stats[4].add_(live.sum()*4096)
            with (worker.run/'prefill-component-timing.jsonl').open('a') as f:
                f.write(json.dumps(dict(shape=signature,first_signature=first,compact_ms=events[0].elapsed_time(events[1]),
                    stock_ms=events[1].elapsed_time(events[2]),scope='Sequential same live inputs; GPU elapsed including preparation and possible JIT gaps; not serving timing'))+'\n')
        return output
    rocm_attn.chunked_prefill_paged_decode=forward
    (worker.run/'prefill-binding.json').write_text(json.dumps(dict(mode=mode,
        source_sha256=hashlib.sha256(pathlib.Path(compact_prefill.__file__).read_bytes()).hexdigest(),
        weight_changes=False,attention_quantization=False,iu4_integer_attention=False,
        scope='Retained BF16 compact varlen prefill, max query >16; native 1056/1072/1088 pages admitted; bounded C1 backing',
        quality_qualified=False,full_optimizer_qualified=False),indent=2)+'\n')
