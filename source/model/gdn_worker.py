"""MiniCPM GDN capture child of the retained shared16 DSpark winner."""
import collections, inspect, json, pathlib, torch
from attention_candidate import Worker as Parent
from gdn_common import bindings, clone_layout, describe, identity

class Worker(Parent):
    def load_model(self,*args,**kwargs):
        result=super().load_model(*args,**kwargs)
        self._gdn_ready=False;self._gdn_concurrency=0
        self._gdn_ordinal=collections.Counter();self._gdn_done=set();self._gdn_records=[]
        sources={}
        for stage,(namespace,key) in bindings().items():
            original=namespace[key];sources[stage]=identity(original)
            signature=inspect.signature(original)
            def capture(*args,_stage=stage,_original=original,_sig=signature,**kwargs):
                output=_original(*args,**kwargs)
                if not self._gdn_ready:return output
                ordinal=self._gdn_ordinal[_stage];self._gdn_ordinal[_stage]+=1
                if ordinal not in (0,29):return output
                bound=_sig.bind(*args,**kwargs);bound.apply_defaults();values=bound.arguments
                t=values['conv_output'].shape[0] if _stage=='postconv' else values['q' if _stage=='chunk_fwd_o' else 'k'].shape[1]
                bucket='long' if t>=4096 else 'tail' if 8<t<=2048 else None
                if bucket is None:return output
                cohort='c1' if self._gdn_concurrency==1 else 'concurrent'
                signature=(_stage,cohort,bucket,ordinal)
                if signature in self._gdn_done:return output
                assert not torch.cuda.is_current_stream_capturing(), 'Actual prefill must not be a synthetic graph capture'
                metadata=dict(stage=_stage,cohort=cohort,bucket=bucket,ordinal=ordinal,rows=t,scheduler_concurrency=self._gdn_concurrency,inputs=describe(values),outputs=describe(output),source=sources[_stage],performance_eligible=False)
                # These three stages do not mutate their inputs except core_attn_out,
                # which is an output-only buffer and is serialized after the call.
                path=self.run/('gdn-operands-'+'-'.join(map(str,signature))+'.pt')
                torch.save(dict(metadata=metadata,arguments=clone_layout(dict(values),'cpu'),expected=clone_layout(output,'cpu')),path)
                metadata['file']=path.name;metadata['bytes']=path.stat().st_size
                self._gdn_records.append(metadata);self._gdn_done.add(signature)
                (self.run/'gdn-capture.json').write_text(json.dumps(self._gdn_records,indent=2)+'\n')
                return output
            namespace[key]=capture
        (self.run/'gdn-binding.json').write_text(json.dumps(dict(mode='actual-operand-capture',sources=sources,scope='Retained parent arithmetic served; synthetic warmup excluded; timing invalid'),indent=2)+'\n')
        return result
    def compile_or_warm_up_model(self):
        result=super().compile_or_warm_up_model();self._gdn_ready=True;return result
    def execute_model(self,scheduler_output):
        self._gdn_concurrency=len(scheduler_output.num_scheduled_tokens)
        self._gdn_ordinal.clear()
        return super().execute_model(scheduler_output)
