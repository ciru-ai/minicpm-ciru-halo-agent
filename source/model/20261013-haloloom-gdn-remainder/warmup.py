import hashlib, importlib, json, os, time
from pathlib import Path
import torch
from kernel_identity import loaded_identity
def sha(value):
    return hashlib.sha256(value).hexdigest()


def install_recorder(root):
    from triton import knobs
    from triton.runtime.jit import JITFunction
    from triton.compiler.compiler import CompiledKernel
    original_compile = JITFunction._do_compile
    original_post = knobs.runtime.jit_post_compile_hook
    original_listener = knobs.compilation.listener
    original_handles = CompiledKernel._init_handles
    destination = (root / 'jit-events.jsonl').open('a', buffering=1)
    def emit(value):
        value['monotonic'] = time.monotonic()
        destination.write(json.dumps(value, default=str)+'\n')
    def post(**kwargs):
        info = kwargs['fn']; fn = info.jit_function
        emit(dict(event='specialization', module=fn.fn.__module__, qualname=fn.fn.__qualname__,
                  source_sha256=sha(fn.src.encode()), dependency_key=fn.cache_key,
                  specialization_data=kwargs['compile'].get('specialization_data')))
        if original_post: return original_post(**kwargs)
    def listener(**kwargs):
        emit(dict(event='compiler', name=kwargs['src'].name,
                  cache_hit=kwargs.get('cache_hit'), times=kwargs.get('times'),
                  hash=kwargs.get('metadata',{}).get('hash')))
        if original_listener: return original_listener(**kwargs)
    def compile(self,*args,**kwargs):
        start=time.monotonic()
        result=original_compile(self,*args,**kwargs)
        emit(dict(event='prepare',name=self._fn_name,wall_s=time.monotonic()-start,
                  hash=result.hash,code_sha256=sha(result.kernel)))
        return result
    def handles(self):
        if self.module is not None: return original_handles(self)
        start=time.monotonic(); result=original_handles(self)
        emit(dict(event='load',name=self.name,hash=self.hash,wall_s=time.monotonic()-start))
        return result
    knobs.runtime.jit_post_compile_hook=post
    knobs.compilation.listener=listener
    JITFunction._do_compile=compile
    CompiledKernel._init_handles=handles



def preload_manifest(path, root):
    import triton
    from triton.runtime.jit import _triton_jit_function_registry
    manifest=json.loads(Path(path).read_text())
    assert manifest['triton_version']==triton.__version__
    before_cpu=torch.get_rng_state().clone(); before_gpu=torch.cuda.get_rng_state().clone()
    started=time.monotonic(); records=[]
    for item in manifest['specializations']:
        identity=item['module']+':'+item['qualname']
        if identity not in _triton_jit_function_registry: importlib.import_module(item['module'])
        fn=_triton_jit_function_registry.get(identity)
        if fn is None: raise RuntimeError('Unresolved warmup function '+identity)
        assert sha(fn.src.encode())==item['source_sha256'], identity+' source changed'
        assert fn.cache_key==item['dependency_key'], identity+' dependencies changed'
        start=time.monotonic(); compiled=fn.preload(item['specialization_data']); prepared=time.monotonic()
        assert compiled.hash==item['expected_hash'], identity+' compile changed'
        assert sha(compiled.kernel)==item['expected_code_sha256'] or loaded_identity(compiled.kernel)==item['expected_loaded_identity'], identity+' GPU code changed'
        compiled._init_handles()
        records.append(dict(name=fn._fn_name,prepare_s=prepared-start,load_s=time.monotonic()-prepared,hash=compiled.hash,code_sha256=sha(compiled.kernel)))
    wanted=set(manifest.get('load_hashes',[])); bound=set(); handles=[]
    for fn in tuple(_triton_jit_function_registry.values()):
        for cache in tuple(fn.device_caches.values()):
            for compiled in tuple(cache[0].values()):
                if getattr(compiled,'hash',None) not in wanted or compiled.hash in bound: continue
                start=time.monotonic(); compiled._init_handles(); bound.add(compiled.hash)
                handles.append(dict(hash=compiled.hash,load_s=time.monotonic()-start))
    assert torch.equal(before_cpu,torch.get_rng_state())
    assert torch.equal(before_gpu,torch.cuda.get_rng_state())
    receipt=dict(manifest=str(path),manifest_sha256=sha(Path(path).read_bytes()),entries=len(records),wall_s=time.monotonic()-started,cpu_rng_unchanged=True,gpu_rng_unchanged=True,model_forward_calls=0,kv_writes=0,records=records,existing_handles=handles,unresolved_handle_hashes=sorted(wanted-bound))
    (root/'serving-warmup.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('MINICPM_SERVING_WARMUP '+json.dumps({k:v for k,v in receipt.items() if k not in ('records','existing_handles')}),flush=True)
