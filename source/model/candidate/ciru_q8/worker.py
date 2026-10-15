"""Initial target binding: retained routed kernels, own original nonexpert weights."""
import ctypes as C, hashlib, json, pathlib
from types import SimpleNamespace
import torch
from vllm.v1.worker.gpu_worker import Worker as VllmGPUWorker
from ornith_g256.worker_base import OrnithWorkerBase
from ornith_g256.lifecycle import WorkerStepLifecycle
from ornith_g256.runtime import Runtime as FlagRuntime
from ornith_g256.ultra_contract import CompleteLayout,NUMERIC_WARMUP_MASK,target_prefix
from .method import CiruQ8MoEMethod

class Binding(FlagRuntime):
    def __init__(self,model,capacity,device,run):
        from ornith_g256 import ultra_native as native
        self.backend=SimpleNamespace(device=torch.device(device));self.capacity=capacity
        modules=[(name,m) for name,m in model.named_modules() if isinstance(getattr(m,'quant_method',None),CiruQ8MoEMethod)]
        assert len(modules)==40 and {target_prefix(m.quant_method.prefix) for _,m in modules}=={f'layers.{i}.mlp.experts' for i in range(40)}
        assert all(m._ornith_weights_ready for _,m in modules)
        self.layers=tuple(m for _,m in modules);self.layer_prefixes=tuple(n for n,_ in modules)
        config=json.loads((pathlib.Path(__file__).resolve().parents[1]/'kernel-bindings.json').read_text())
        self.banks={};self.layout={}
        for name,row in config['libraries'].items():
            path=pathlib.Path(row['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
            lib=C.CDLL(str(path));q=getattr(lib,row['entry_symbol'].removesuffix('_launch')+'_get_layout')
            q.argtypes,q.restype=[C.c_int]*5+[C.POINTER(CompleteLayout)],C.c_int
            fn=getattr(lib,row['entry_symbol']);fn.argtypes=[C.c_void_p]*10+[C.c_size_t]+[C.c_void_p]*2+[C.c_int]*6+[C.c_void_p];fn.restype=C.c_int
            layout=CompleteLayout();cap=capacity if name=='prefill' else config['small_layout_capacity']
            assert q(cap,256,2048,512,8,C.byref(layout))==0
            self.banks[name]=(row,lib,q,fn);self.layout[name]=layout
        self.workspace=torch.empty(max(l.workspace_bytes for l in self.layout.values()),dtype=torch.uint8,device=device)
        assert self.workspace.data_ptr()%256==0
        self.error_flags=torch.zeros(40,dtype=torch.int32,device=device);self.slots=tuple(self.error_flags[i:i+1] for i in range(40))
        self.seen=set()
        def entry(prefill,*args):
            args=list(args);rows=args[13];assert 0<rows<=args[14]<=capacity
            name='prefill' if prefill else 'small' if rows<=32 else 'small_fallback'
            if not prefill:args[14]=config['small_layout_capacity']
            row,lib,q,fn=self.banks[name];status=fn(*args)
            key=(name,rows,args[14])
            if key not in self.seen:
                self.seen.add(key)
                with (run/'native-calls.jsonl').open('a') as f:f.write(json.dumps(dict(kernel=name,rows=rows,capacity=args[14],status=status,
                    path=row['path'],sha256=row['sha256'],symbol=row['entry_symbol'],capture=torch.cuda.is_current_stream_capturing(),
                    scope='Native host invocation; hardware dispatch join and graph replay collector unqualified'))+'\n')
            return status
        def native_tuple(name,call,threshold=None):
            row,lib,q,fn=self.banks[name];identity=(row['path'],row['sha256'],row['entry_symbol'])
            if threshold is not None:identity+=(threshold,)
            return identity,lib,q,call
        native._Q8=native_tuple('small',lambda *a:entry(False,*a))
        native._Q8_PREFILL=native_tuple('prefill',lambda *a:entry(True,*a),config['prefill_min_rows'])
        for i,m in enumerate(self.layers):m.quant_method.runtime=self;m.quant_method.slot=i
        (run/'target-binding.json').write_text(json.dumps(dict(status='TARGET_ROUTED_BOUND',routed_layers=40,
            kernels=config,arena_bytes=self.workspace.numel(),native_payload_bytes=sum(p.numel()*p.element_size() for m in self.layers for p in m.parameters(recurse=False)),
            nonexpert_policy='Own original BF16, unchanged vLLM operators at this initial target checkpoint',
            speculation_qualified=False,quality_qualified=False,full_optimizer_qualified=False),indent=2)+'\n')

class Worker(OrnithWorkerBase):
    def __init__(self,*args,**kwargs):
        self._ornith_binding=self._ornith_lifecycle=self._ornith_pending=None
        super().__init__(*args,**kwargs)
        c=self.vllm_config
        assert not c.scheduler_config.async_scheduling and c.scheduler_config.max_num_seqs<=8
        assert c.scheduler_config.max_num_batched_tokens<=8192
        if type(self) is Worker:assert c.speculative_config is None
        self.run=pathlib.Path(c.additional_config['ciru_q8']['run']);self.run.mkdir(parents=True,exist_ok=True)
    def load_model(self,*args,**kwargs):
        result=VllmGPUWorker.load_model(self,*args,**kwargs)
        before=torch.cuda.memory_allocated(self.device)
        self._ornith_binding=Binding(self.model_runner.get_model(),self.vllm_config.scheduler_config.max_num_batched_tokens,self.device,self.run)
        prefill=self.vllm_config.additional_config['ciru_q8'].get('prefill','stock')
        if prefill!='stock':
            from .prefill_bridge import install
            install(self,prefill)
        mode=self.vllm_config.additional_config['ciru_q8'].get('attention','stock')
        if mode!='stock':
            from .attention_bridge import install
            install(self,mode)
        self.model_runner.model_memory_usage+=torch.cuda.memory_allocated(self.device)-before
        self._ornith_lifecycle=WorkerStepLifecycle(self._ornith_binding,self.vllm_config.max_concurrent_batches+1)
        return result
    def compile_or_warm_up_model(self):
        lifecycle=self._require_lifecycle();ticket=lifecycle.begin('synthetic-warmup')
        try:
            result=VllmGPUWorker.compile_or_warm_up_model(self)
            flags=self._ornith_binding.error_flags.cpu().tolist()
            for i,code in enumerate(flags):
                if code and code&~NUMERIC_WARMUP_MASK==0:self._ornith_binding.error_flags[i]=0
            (self.run/'synthetic-warmup-flags.json').write_text(json.dumps(flags)+'\n')
            lifecycle.queue(ticket);lifecycle.complete(ticket)
            if self.vllm_config.additional_config['ciru_q8'].get('attention','stock')!='stock':
                assert self._attention_dispatch_seen,'Retained attention was requested but no eligible capture dispatched'
            if getattr(self,'_attention_stats',None) is not None:self._attention_stats.zero_()
            self._prefill_real=True
            if getattr(self,'_prefill_stats',None) is not None:self._prefill_stats.zero_()
            return result
        except Exception as error:lifecycle.abort(ticket,error)
    def _save_attention(self):
        if getattr(self,'_prefill_stats',None) is not None:
            v=self._prefill_stats.cpu().tolist()
            (self.run/'prefill-actual-operand-comparison.json').write_text(json.dumps(dict(real_dispatches=v[0],max_absolute_error=v[1],max_per_dispatch_rms_error=v[2],nonfinite_values=v[3],compared_values=v[4],scope='Stock and compact BF16 prefill on identical live inputs, padding excluded; audit serves candidate; timing receipts include preparation.'),indent=2)+'\n')
        if getattr(self,'_attention_stats',None) is not None:
            values=self._attention_stats.cpu().tolist()
            pending=self.run/'attention-actual-operand-comparison.json.tmp'
            pending.write_text(json.dumps(dict(
                real_dispatches=values[0],max_absolute_error=values[1],max_per_dispatch_rms_error=values[2],
                nonfinite_values=values[3],compared_values=values[4],
                scope='Both operators receive identical actual live Q/K/V/cache metadata, including graph replay. Padding excluded. Startup dummy statistics reset before API readiness. Snapshot after each real sampling step; no reliance on process shutdown. Reference stock attention; served output retained partitioned candidate. Diagnostic overhead excluded from clean speed.'),indent=2)+'\n')
            pending.replace(self.run/'attention-actual-operand-comparison.json')
    def execute_model(self,scheduler_output):
        result=super().execute_model(scheduler_output)
        if result is not None:self._save_attention()
        return result
    def sample_tokens(self,grammar_output):
        result=super().sample_tokens(grammar_output);self._save_attention();return result
    def shutdown(self):
        self._save_attention()
        return super().shutdown()
