import json, os, time
from pathlib import Path
from bf16_prefill_worker import Worker as Parent
from warmup import install_recorder, preload_manifest
class Worker(Parent):
    def compile_or_warm_up_model(self):
        result=super().compile_or_warm_up_model()
        root=Path(os.environ['CIRU_SERVING_RUN']); mode=os.environ['CIRU_SERVING_MODE']
        if mode in ('record','candidate-record'): install_recorder(root)
        if mode in ('candidate','candidate-record'):
            preload_manifest(os.environ['CIRU_SERVING_MANIFEST'],root)
        (root/'worker-ready.json').write_text(json.dumps(dict(mode=mode,monotonic=time.monotonic(),pid=os.getpid()))+'\n')
        return result
