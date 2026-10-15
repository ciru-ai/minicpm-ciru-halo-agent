"""Integrated depth dispatch; no diagnostic controls or GPU timing hooks."""
import hashlib
import inspect
import json
import os
from pathlib import Path
from serving_worker import Worker as Parent


class Worker(Parent):
    def load_model(self, *args, **kwargs):
        value = super().load_model(*args, **kwargs)
        if os.environ.get('CIRU_DEPTH_ADAPTIVE') == '1':
            from adaptive_dspark import install
            install(self)
        return value

    def compile_or_warm_up_model(self):
        value = super().compile_or_warm_up_model()
        r = self.model_runner
        modules = [inspect.getmodule(r), inspect.getmodule(r.speculator), inspect.getmodule(r.cudagraph_manager)]
        files = [Path(m.__file__) for m in modules] + [Path(__file__), Path(__file__).with_name('adaptive_dspark.py')]
        (self.run/'adaptive-source-binding.json').write_text(json.dumps(dict(
            sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            adaptive=os.environ.get('CIRU_DEPTH_ADAPTIVE') == '1',
            graphs=[str(g) for g in r.cudagraph_manager.graphs],
            worker='release_worker.Worker', diagnostic_hooks=False,
            weights_changed=False, full_optimizer_qualified=False), indent=2)+'\n')
        return value
