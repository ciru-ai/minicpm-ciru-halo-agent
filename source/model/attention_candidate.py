"""Isolated target-only BF16 attention child; retained candidate source untouched."""
import hashlib,inspect,json,os,pathlib,torch
from ciru_q8.dspark_worker import Worker as Parent
class Worker(Parent):
    def load_model(self,*args,**kwargs):
        result=super().load_model(*args,**kwargs)
        from ciru_q8 import attention_bridge
        import attention_shared_query
        original=attention_bridge.partition
        arm=os.environ['CIRU_ATTENTION_CHILD'];assert arm in ('row64','shared16','shared64')
        audit=os.environ.get('CIRU_ATTENTION_CHILD_AUDIT','0')=='1'
        self._attention_child_stats=torch.zeros(5,device=self.device,dtype=torch.float32) if audit else None
        if arm=='row64':
            source=inspect.getsource(original);assert source.count('BLOCK_N=16')==1
            namespace=dict(original.__globals__);exec(compile(source.replace('BLOCK_N=16','BLOCK_N=64'),__file__+'.row64.generated','exec'),namespace);selected=namespace['forward']
        else:
            def selected(*args,**kwargs):return attention_shared_query.forward(*args,**kwargs,block_n=int(arm.removeprefix('shared')))
        seen=set()
        def call(q,k,v,out,table,starts,seq,scale,ks,vs,**options):
            # No claimed query-sharing benefit for a single query; preserve it.
            use=arm=='row64' or options['max_query_len']>1
            if not use:return original(q,k,v,out,table,starts,seq,scale,ks,vs,**options)
            selected(q,k,v,out,table,starts,seq,scale,ks,vs,**options)
            stats=self._attention_child_stats
            if stats is not None:
                ref=torch.empty_like(out);original(q,k,v,ref,table,starts,seq,scale,ks,vs,**options)
                live=torch.arange(q.shape[0],device=q.device)<starts[-1];active=live[:,None,None]
                diff=torch.where(active,out.float()-ref.float(),0.)
                stats[0].add_(live.any());stats[1].copy_(torch.maximum(stats[1],diff.abs().nan_to_num().amax()))
                rms=(diff.square().sum()/(live.sum()*4096).clamp_min(1)).sqrt()
                stats[2].copy_(torch.maximum(stats[2],rms.nan_to_num()))
                stats[3].add_((active&~torch.isfinite(out)).sum()+(active&~torch.isfinite(ref)).sum());stats[4].add_(live.sum()*4096)
            sig=(q.shape[0],seq.numel(),options['max_query_len'],k.shape[3],torch.cuda.is_current_stream_capturing())
            if sig not in seen:
                seen.add(sig)
                with (self.run/'attention-child-calls.jsonl').open('a') as f:f.write(json.dumps(dict(arm=arm,signature=sig,query_stride=list(q.stride()),key_stride=list(k.stride()),value_stride=list(v.stride()),output_stride=list(out.stride()),audit=audit,scope='Host invocation and capture; audit statistics include real graph replay'))+'\n')
            return out
        attention_bridge.partition=call
        paths=[pathlib.Path(__file__),pathlib.Path(attention_shared_query.__file__),pathlib.Path(inspect.getfile(original))]
        (self.run/'attention-child-binding.json').write_text(json.dumps(dict(arm=arm,audit=audit,weights_changed=False,cache_values_and_layout_changed=False,target_only=True,parent='candidate/ciru_q8/attention_partition.py',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},quality_qualified=False,full_optimizer_qualified=False),indent=2)+'\n')
        return result
    def compile_or_warm_up_model(self):
        result=super().compile_or_warm_up_model()
        if self._attention_child_stats is not None:self._attention_child_stats.zero_()
        return result
    def _save_attention(self):
        super()._save_attention()
        stats=getattr(self,'_attention_child_stats',None)
        if stats is not None and not getattr(self,'_dspark_warmup',False):
            v=stats.cpu().tolist();(self.run/'attention-child-actual-operands.json').write_text(json.dumps(dict(real_dispatches=v[0],max_abs=v[1],max_dispatch_rms=v[2],nonfinite=v[3],compared_values=v[4],scope='Target child and retained attention on identical live operands including graph replay; synthetic statistics reset; audit serves child; timing excluded.'),indent=2)+'\n')
