"""Bind one immutable routed child before warmup; retain the model60 runtime."""
import ctypes as C
import hashlib
import json
from pathlib import Path

from remainder_worker import Worker as Parent

HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'candidates/gu64k-producer-v3/complete-manifest.json'
MANIFEST_SHA='9f5f3eea338f5eb72616c494ed0962135dacb7fd210d1db6c16d7dabe583957e'


def bundle_pins():
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(MANIFEST)==MANIFEST_SHA
    manifest=json.loads(MANIFEST.read_text())
    values={str(MANIFEST):MANIFEST_SHA,str(Path(__file__)):sha(__file__)}
    for p,entry in manifest['sources'].items():
        assert sha(p)==entry['sha256'],p
        values[p]=entry['sha256']
    for entry in manifest['libraries']:
        assert sha(entry['path'])==entry['sha256']
        values[entry['path']]=entry['sha256']
    return values


class Worker(Parent):
    def load_model(self,*args,**kwargs):
        pins=bundle_pins()
        result=super().load_model(*args,**kwargs)
        from ornith_g256 import ultra_native
        from ornith_g256.ultra_contract import CompleteLayout
        binding=self._ornith_binding
        assert not binding.seen,'Candidate must bind before any routed invocation or graph capture'
        old=ultra_native._Q8_PREFILL
        manifest=json.loads(MANIFEST.read_text());entry=manifest['libraries'][0]
        lib=C.CDLL(entry['path'])
        query=getattr(lib,manifest['layout_symbol'])
        query.argtypes,query.restype=[C.c_int]*5+[C.POINTER(CompleteLayout)],C.c_int
        launch=getattr(lib,entry['entry_symbol'])
        launch.argtypes,launch.restype=[C.c_void_p]*10+[C.c_size_t]+[C.c_void_p]*2+[C.c_int]*6+[C.c_void_p],C.c_int
        layout=CompleteLayout()
        assert query(binding.capacity,256,2048,512,8,C.byref(layout))==0
        parent_layout=binding.layout['prefill']
        assert all(getattr(layout,k)==getattr(parent_layout,k) for k,_ in CompleteLayout._fields_)
        binding.banks['prefill']=(entry,lib,query,launch)
        binding.layout['prefill']=layout
        identity=(entry['path'],entry['sha256'],entry['entry_symbol'],old[0][3])
        # The existing entry closure now dispatches the new bank and records it.
        ultra_native._Q8_PREFILL=(identity,lib,query,old[3])
        receipt=dict(status='BOUND_BEFORE_WARMUP',parent_identity=old[0],candidate_identity=identity,
            manifest=str(MANIFEST),pins=pins,workspace_layout_unchanged=True,
            arena_bytes=binding.workspace.numel(),layers=len(binding.layers),
            small_decode_unchanged=True,weights_changed=False,source_edit_candidate=True,
            function_address=C.cast(launch,C.c_void_p).value,full_optimizer_qualified=False)
        (self.run/'routed-candidate-binding.json').write_text(json.dumps(receipt,indent=2)+'\n')
        path=self.run/'target-binding.json';target=json.loads(path.read_text())
        target['retained_kernels']=json.loads(json.dumps(target['kernels']))
        target['kernels']['libraries']['prefill']=entry
        target['source_edit_candidate']=receipt
        path.write_text(json.dumps(target,indent=2)+'\n')
        return result


from profile_worker import Worker as ProfilingParent


class ProfileWorker(ProfilingParent,Worker):
    """Same binding with the existing selected-region profiler."""
