"""Three measured Ornith traversal changes; preserve all target/draft arithmetic."""
import collections, inspect, json, os, torch
from attention_candidate import Worker as Parent
from gdn_common import bindings, describe, identity
from ornith_g256.optimized.pipeline.adapter import install, ParentConfigurationRequired
from ornith_g256.optimized.postconv import fused_post_conv_prep

class Worker(Parent):
    def load_model(self,*args,**kwargs):
        value=super().load_model(*args,**kwargs)
        self._gdn_ready=False
        self._gdn_audit=os.environ.get('CIRU_GDN_MODE')=='audit'
        self._gdn_profile=os.environ.get('CIRU_GDN_PROFILE')=='1'
        self._gdn_pipe=install(lambda:False);self._gdn_pipe.restore()
        self._gdn_stats={};self._gdn_calls=collections.Counter();self._gdn_seen=set();self._gdn_dispatches=[]
        sources={};self._gdn_roctx=None
        for stage,(namespace,key) in bindings().items():
            original=namespace[key]
            selected=fused_post_conv_prep if stage=='postconv' else self._gdn_pipe.candidate[stage]
            sources[stage]=dict(parent=identity(original),child=identity(selected))
            if stage=='postconv':assert sources[stage]['parent']['sha256']=='590717b2107abc90d72548de06b5d315229eb2775c94614f3f545df11cc2b5df'
            signature=inspect.signature(original)
            if self._gdn_audit:self._gdn_stats[stage]=torch.zeros(3,dtype=torch.float64,device=self.device)
            def call(*args,_stage=stage,_original=original,_selected=selected,_sig=signature,**kwargs):
                if not self._gdn_ready:return _original(*args,**kwargs)
                bound=_sig.bind(*args,**kwargs);bound.apply_defaults();values=dict(bound.arguments)
                rows=values['conv_output'].shape[0] if _stage=='postconv' else values['q' if _stage=='chunk_fwd_o' else 'k'].shape[1]
                if rows<=8:return _original(*args,**kwargs)
                trace=self._gdn_roctx is not None and rows>=4096 and _stage not in self._gdn_profiled
                if trace:
                    torch.cuda.synchronize();self._gdn_roctx.resume();depth=self._gdn_roctx.push('GDN_SPATIAL_'+_stage)
                try:
                    try:output=_selected(*args,**kwargs)
                    except ParentConfigurationRequired:
                        self._gdn_calls[_stage+'_tuner_fallback']+=1
                        return _original(*args,**kwargs)
                finally:
                    if trace:
                        torch.cuda.synchronize();self._gdn_roctx.pop(depth);self._gdn_roctx.pause()
                self._gdn_calls[_stage]+=1
                if trace:
                    self._gdn_profiled.add(_stage)
                    self._gdn_dispatches.append(dict(stage=_stage,marker='GDN_SPATIAL_'+_stage,inputs=describe(values),sources=sources[_stage]))
                    (self.run/'gdn-native-selected.json').write_text(json.dumps(self._gdn_dispatches,indent=2)+'\n')
                if self._gdn_audit:
                    # chunkO can write an output buffer; keep the reference buffer
                    # separate so that the audit cannot overwrite served output.
                    if values.get('core_attn_out') is not None:
                        v=values['core_attn_out'];values['core_attn_out']=torch.empty_strided(v.shape,v.stride(),dtype=v.dtype,device=v.device)
                    reference=_original(**values)
                    aa=output if isinstance(output,(tuple,list)) else (output,)
                    bb=reference if isinstance(reference,(tuple,list)) else (reference,)
                    stats=self._gdn_stats[_stage]
                    for a,b in zip(aa,bb,strict=True):
                        stats[0].add_((a!=b).sum());stats[1].add_((~a.isfinite()).sum()+(~b.isfinite()).sum())
                        stats[2].copy_(torch.maximum(stats[2],(a.float()-b.float()).abs().amax().double()))
                        self._gdn_calls[_stage+'_values']+=a.numel()
                    sig=(_stage,rows,str(describe(values)))
                    if sig not in self._gdn_seen:
                        self._gdn_seen.add(sig)
                        with (self.run/'gdn-actual-shapes.jsonl').open('a') as f:f.write(json.dumps(dict(stage=_stage,inputs=describe(values)))+'\n')
                return output
            namespace[key]=call
        (self.run/'gdn-binding.json').write_text(json.dumps(dict(mode='audit' if self._gdn_audit else 'clean',sources=sources,weights_changed=False,precision_changed=False,arithmetic_changed=False,decode_rows_le8_unchanged=True,scope='Reuse measured Ornith traversal for postconv/WU/chunkO only. Parent autotuner cache/configs shared, unseen keys fall back to parent. Convolution and recurrent state update unchanged.',full_optimizer_qualified=False),indent=2)+'\n')
        return value
    def compile_or_warm_up_model(self):
        result=super().compile_or_warm_up_model()
        if self._gdn_profile:
            from dspark_profile import Roctx
            self._gdn_roctx=Roctx();self._gdn_roctx.pause();self._gdn_profiled=set()
        self._gdn_ready=True
        return result
    def _save_attention(self):
        super()._save_attention()
        if not getattr(self,'_gdn_ready',False):return
        if self._gdn_audit:
            stats={k:v.cpu().tolist() for k,v in self._gdn_stats.items()}
            (self.run/'gdn-actual-operand-audit.json').write_text(json.dumps(dict(stats=stats,columns=['unequal_values','nonfinite_values','max_abs'],calls=dict(self._gdn_calls),parent_configurations=self._gdn_pipe.snapshot(),scope='All affected live layers on identical operands; synthetic warmup excluded; candidate served. Timing invalid.'),indent=2)+'\n')
            assert all(x==[0.,0.,0.] for x in stats.values()),stats
