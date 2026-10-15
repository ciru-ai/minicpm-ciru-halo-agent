"""Selected live V2 target/sample/draft GPU regions; never clean timing evidence.

Uses the established Ciru rocprofv3/ROCTX selected-region route. No math,
sampling, graph mode, cache, scheduler or error-lifecycle changes.
"""
import copy, ctypes as C, functools, json, pathlib, time
import torch
from ciru_q8.dspark_worker import Worker as Parent

class Roctx:
    def __init__(self):
        self.lib=C.CDLL('/opt/ciru/glm53-iu4/venv/lib/python3.14/site-packages/_rocm_sdk_devel/lib/librocprofiler-sdk-roctx.so.1')
        for name in ['roctxProfilerPause','roctxProfilerResume']:
            f=getattr(self.lib,name);f.argtypes=[C.c_uint64];f.restype=C.c_int
        self.lib.roctxRangePushA.argtypes=[C.c_char_p];self.lib.roctxRangePushA.restype=C.c_int
        self.lib.roctxRangePop.argtypes=[];self.lib.roctxRangePop.restype=C.c_int
    def pause(self):assert self.lib.roctxProfilerPause(0)==0
    def resume(self):assert self.lib.roctxProfilerResume(0)==0
    def push(self,label):
        depth=self.lib.roctxRangePushA(label.encode());assert depth>=0;return depth
    def pop(self,depth):assert self.lib.roctxRangePop()==depth

class Worker(Parent):
    def compile_or_warm_up_model(self):
        value=super().compile_or_warm_up_model()
        self._profile_roctx=Roctx();self._profile_roctx.pause()
        self._profile_rows=[];self._profile_pending=None;self._profile_skipped={}
        r=self.model_runner
        for name in ['prepare_inputs','prepare_attn','sample','postprocess_sampled','postprocess_num_computed_tokens']:
            self._wrap(r,name)
        self._wrap(r.get_model(),'compute_logits','target_head')
        if r.speculator is not None:self._wrap(r.speculator,'propose','draft_propose')
        # Shape/weight inventory only; no numerical export or GPU copies.
        inventory=[]
        for owner,model in [('target',r.get_model()),('draft',r.speculator.model if r.speculator else None)]:
            if model is None:continue
            for name,module in model.named_modules():
                weight=getattr(module,'weight',None)
                if isinstance(weight,torch.Tensor) and weight.ndim==2:
                    inventory.append(dict(owner=owner,name=name,shape=list(weight.shape),dtype=str(weight.dtype),bytes=weight.numel()*weight.element_size(),method=type(getattr(module,'quant_method',None)).__name__))
        (self.run/'linear-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
        config=copy.copy(self.vllm_config);config.model_config=copy.copy(config.model_config)
        capacities={}
        for length in [63875+256,65536,262144]:
            config.model_config.max_model_len=length
            capacities[str(length)]=get_max_concurrency_for_kv_cache_config(config,r.kv_cache_config)
        (self.run/'cache-capacity.json').write_text(json.dumps(dict(num_blocks=r.kv_cache_config.num_blocks,kv_bytes=self.vllm_config.cache_config.kv_cache_memory_bytes,max_concurrency_at_length=capacities,scope='Native capacity estimate; live cache eviction still requires counters'),indent=2)+'\n')
        self._save_profile();return value
    def _wrap(self,owner,name,label=None):
        if not hasattr(owner,name):return
        original=getattr(owner,name)
        @functools.wraps(original)
        def call(*args,**kwargs):
            row=self._profile_pending
            if row is None:return original(*args,**kwargs)
            marker=f'MCP_{row["shape"]}_{row["index"]}_{label or name}'
            depth=self._profile_roctx.push(marker);start=time.monotonic_ns()
            try:return original(*args,**kwargs)
            finally:
                self._profile_roctx.pop(depth)
                row['host_stages'].append(dict(name=label or name,marker=marker,duration_ns=time.monotonic_ns()-start))
        setattr(owner,name,call)
    def _save_profile(self):
        counts={shape:sum(r['shape']==shape for r in self._profile_rows) for shape in ['c1','c8','prefill']}
        status='COMPLETE' if counts=={'c1':8,'c8':8,'prefill':1} else 'CAPTURING'
        (self.run/'live-profile.json').write_text(json.dumps(dict(status=status,counts=counts,steps=self._profile_rows,diagnostic_only=True,performance_eligible=False,quality_evidence=False,full_optimizer_qualified=False),indent=2)+'\n')
    def execute_model(self,scheduler_output,*args,**kwargs):
        if not hasattr(self,'_profile_roctx'):return super().execute_model(scheduler_output,*args,**kwargs)
        counts=scheduler_output.num_scheduled_tokens;drafts=scheduler_output.scheduled_spec_decode_tokens
        total=sum(counts.values());c=len(counts);shape=None
        if c in (1,8) and all(n==3 and len(drafts.get(k,[]))==2 for k,n in counts.items()):shape='c1' if c==1 else 'c8'
        elif c==1 and total>=4096 and not any(drafts.values()):shape='prefill'
        seen=sum(r['shape']==shape for r in self._profile_rows)
        limit=1 if shape=='prefill' else 8
        if shape is None or seen>=limit:return super().execute_model(scheduler_output,*args,**kwargs)
        skipped=self._profile_skipped.get(shape,0)
        if skipped<2:
            self._profile_skipped[shape]=skipped+1
            return super().execute_model(scheduler_output,*args,**kwargs)
        assert self._profile_pending is None
        cached=scheduler_output.scheduled_cached_reqs
        row=dict(shape=shape,index=seen,scheduled_tokens=total,requests=c,host_stages=[],worker_pid=__import__('os').getpid(),request_ids=list(counts),cached_computed=list(map(int,cached.num_computed_tokens)),cached_output=list(map(int,cached.num_output_tokens)))
        self._profile_pending=row
        marker=f'MCP_{shape}_{seen}_TARGET';row['target_marker']=marker
        torch.cuda.synchronize(self.device);self._profile_roctx.resume();depth=self._profile_roctx.push(marker)
        start=time.monotonic_ns()
        try:
            value=super().execute_model(scheduler_output,*args,**kwargs)
            torch.cuda.synchronize(self.device)
        finally:self._profile_roctx.pop(depth);self._profile_roctx.pause()
        row['target_wall_ms']=(time.monotonic_ns()-start)/1e6
        if value is not None:
            self._profile_rows.append(row);self._profile_pending=None;self._save_profile()
        return value
    def sample_tokens(self,*args,**kwargs):
        row=getattr(self,'_profile_pending',None)
        if row is None:return super().sample_tokens(*args,**kwargs)
        marker=f'MCP_{row["shape"]}_{row["index"]}_SAMPLE_DRAFT';row['sample_marker']=marker
        torch.cuda.synchronize(self.device);self._profile_roctx.resume();depth=self._profile_roctx.push(marker)
        start=time.monotonic_ns()
        try:
            value=super().sample_tokens(*args,**kwargs)
            torch.cuda.synchronize(self.device)
        finally:self._profile_roctx.pop(depth);self._profile_roctx.pause()
        row['sample_wall_ms']=(time.monotonic_ns()-start)/1e6
        self._profile_rows.append(row);self._profile_pending=None;self._save_profile()
        return value
