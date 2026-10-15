"""Canonicalize an unused compile key and prepare actual-layout attention code."""
import hashlib
from pathlib import Path
import time

import torch
import attention_shared_query as shared
import attention_w8 as warp8


def canonical_forward():
    path = Path(shared.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '3e19c89ba36809c82c820c4dbebbb9efc43286fbbac9252632dc4ecca437c23c'
    source = path.read_text().split('\ndef forward(', 1)[1]
    source = 'def forward(' + source
    assert source.count('num_warps=4,num_stages=1') == source.count('ROWS=rows') == 1
    source = source.replace('num_warps=4,num_stages=1', 'num_warps=8,num_stages=1').replace('ROWS=rows', 'ROWS=0')
    namespace = dict(warp8.forward.__globals__)
    exec(compile(source, __file__+'.forward', 'exec'), namespace)
    return namespace['forward']


class Templates:
    """Observe graph-setup operands; restore original run methods before serving."""
    def __init__(self):
        self.saved = {}
        self.templates = {}
        for name in ('shared_query_stage1', 'shared_query_reduce'):
            fn = getattr(shared, name)
            original = fn.run
            self.saved[name] = original

            def observe(*args, _name=name, _original=original, **kwargs):
                kernel = _original(*args, **kwargs)
                if _name not in self.templates:
                    self.templates[_name] = (args, dict(kwargs))
                return kernel

            fn.run = observe

    def close(self):
        for name, original in self.saved.items():
            getattr(shared, name).run = original

    def prepare(self, verify=True):
        assert set(self.templates) == set(self.saved)
        before_cpu = torch.get_rng_state().clone()
        before_gpu = torch.cuda.get_rng_state().clone()
        rows = []
        started = time.monotonic()
        for name, values in (('shared_query_stage1', range(1,17)), ('shared_query_reduce', range(1,9))):
            args, template = self.templates[name]
            for value in values:
                kw = dict(template, warmup=True)
                if name == 'shared_query_stage1':
                    kw['MAX_Q'] = value
                else:
                    kw.update(ROWS=0, REQS=value, RB=1 << (value-1).bit_length())
                start = time.monotonic()
                fn = getattr(shared, name)
                # Compile-only preparation must not pass through the retained
                # live-launch recorder, which expects initialized register data.
                compiled = type(fn).run(fn, *args, **kw)
                compiled._init_handles()
                duration = time.monotonic()-start
                digest = hashlib.sha256(compiled.asm['hsaco']).hexdigest()
                item = dict(kernel=name, value=value, wall_s=duration, compiler_hash=compiled.hash,
                    hsaco_sha256=digest, constants={k:v for k,v in kw.items() if k not in ('grid','warmup')},
                    tensor_layouts=[dict(shape=list(a.shape),stride=list(a.stride()),dtype=str(a.dtype),
                                        address_mod16=a.data_ptr()%16) for a in args])
                if name == 'shared_query_reduce' and verify:
                    original = type(fn).run(fn, *args, **dict(kw, ROWS=args[1].shape[0]))
                    item['original_rows'] = args[1].shape[0]
                    item['original_compiler_hash'] = original.hash
                    item['same_gpu_binary'] = original.asm['hsaco'] == compiled.asm['hsaco']
                    assert item['same_gpu_binary'], item
                rows.append(item)
        assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_gpu,torch.cuda.get_rng_state())
        return dict(entries=rows, wall_s=time.monotonic()-started, model_forward_calls=0,
            kernel_launches=0, kv_writes=0, cpu_rng_unchanged=True, gpu_rng_unchanged=True,
            scope='Compile and initialize handles using actual graph-layout operands; no GPU kernel launch.')
