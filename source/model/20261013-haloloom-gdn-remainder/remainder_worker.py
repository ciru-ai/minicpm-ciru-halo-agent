"""Add the two measured prefill traversal changes to the integrated model."""
import collections
import inspect
import json
import os
import time

import torch
from release_worker import Worker as Parent
from gdn_common import describe, identity
from ornith_g256.optimized.pipeline.adapter import ParentConfigurationRequired


class Worker(Parent):
    def load_model(self, *args, **kwargs):
        value=super().load_model(*args,**kwargs)
        from vllm.third_party.flash_linear_attention.ops import chunk
        namespace=chunk.chunk_gated_delta_rule_fwd.__globals__
        self._remainder_ready=False
        self._remainder_audit=os.environ.get('CIRU_REMAINDER_MODE')=='audit'
        self._remainder_calls=collections.Counter()
        self._remainder_seen=set()
        self._remainder_stats={}
        self._remainder_dirty=False
        sources={}
        for stage in ('chunk_scaled_dot_kkt_fwd','solve_tril'):
            original=namespace[stage]
            assert original is self._gdn_pipe.original[stage],stage
            selected=self._gdn_pipe.candidate[stage]
            signature=inspect.signature(original)
            sources[stage]=dict(parent=identity(original),child=identity(selected))
            if self._remainder_audit:
                self._remainder_stats[stage]=torch.zeros(3,dtype=torch.float64,device=self.device)

            def call(*args,_stage=stage,_original=original,_selected=selected,_signature=signature,**kwargs):
                if not self._remainder_ready:
                    return _original(*args,**kwargs)
                bound=_signature.bind(*args,**kwargs);bound.apply_defaults();values=dict(bound.arguments)
                operand=values['A' if _stage=='solve_tril' else 'k']
                rows=operand.shape[1]
                if rows<=8 or (_stage=='solve_tril' and operand.shape[-1]!=64):
                    return _original(*args,**kwargs)
                try:
                    result=_selected(*args,**kwargs)
                except ParentConfigurationRequired:
                    self._remainder_calls[_stage+'_tuner_fallback']+=1
                    return _original(*args,**kwargs)
                self._remainder_calls[_stage]+=1
                self._remainder_dirty=True
                key=(_stage,rows,str(describe(values)))
                if key not in self._remainder_seen:
                    self._remainder_seen.add(key)
                    with (self.run/'remainder-dispatch.jsonl').open('a') as f:
                        f.write(json.dumps(dict(stage=_stage,rows=rows,host_monotonic_ns=time.monotonic_ns(),inputs=describe(values),sources=sources[_stage],path='retained spatial candidate, parent-selected tuner configuration'))+'\n')
                if self._remainder_audit:
                    reference=_original(**values)
                    stats=self._remainder_stats[_stage]
                    stats[0].add_((result!=reference).sum())
                    stats[1].add_((~result.isfinite()).sum()+(~reference.isfinite()).sum())
                    stats[2].copy_(torch.maximum(stats[2],(result.float()-reference.float()).abs().amax().double()))
                    self._remainder_calls[_stage+'_values']+=result.numel()
                return result

            namespace[stage]=call
        (self.run/'remainder-binding.json').write_text(json.dumps(dict(worker=type(self).__module__+'.'+type(self).__name__,mode='audit' if self._remainder_audit else 'clean',sources=sources,weights_changed=False,arithmetic_changed=False,adaptive_drafts_unchanged=True,baseline='integrated model60',full_optimizer_qualified=False),indent=2)+'\n')
        return value

    def compile_or_warm_up_model(self):
        result=super().compile_or_warm_up_model()
        self._remainder_ready=True
        return result

    def _save_attention(self):
        super()._save_attention()
        if not getattr(self,'_remainder_ready',False) or not self._remainder_dirty:
            return
        stats={k:v.cpu().tolist() for k,v in self._remainder_stats.items()}
        (self.run/'remainder-actual-operands.json').write_text(json.dumps(dict(mode='audit' if self._remainder_audit else 'clean',stats=stats,columns=['unequal_values','nonfinite_values','max_abs'],calls=dict(self._remainder_calls),parent_configurations=self._gdn_pipe.snapshot(),scope='Audit compares every affected live call against original on identical operands. Clean mode records dispatch only.'),indent=2)+'\n')
        self._remainder_dirty=False
        assert all(x==[0.,0.,0.] for x in stats.values()),stats
