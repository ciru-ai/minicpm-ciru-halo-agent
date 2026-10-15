"""One candidate delta: reuse retained GU on prefill rows256..1119."""
import hashlib
import json
from pathlib import Path
from model60_latest_worker import Worker as Parent

class Worker(Parent):
    def load_model(self,*args,**kwargs):
        value=super().load_model(*args,**kwargs)
        from ornith_g256 import ultra_native
        assert not self._ornith_binding.seen
        old=ultra_native._Q8_PREFILL
        assert old[0][3]==1120
        identity=(*old[0][:3],256)
        ultra_native._Q8_PREFILL=(identity,*old[1:])
        (self.run/'small-prefill-candidate-binding.json').write_text(json.dumps(dict(
            delta='Prefill bank minimum rows1120 to256 only',before=list(old[0]),after=list(identity),
            worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            weights_changed=False,decode_rows_1_to_32_unchanged=True,rows_33_to_255_unchanged=True,
            long_prefill_bank_unchanged=True,graph_capture_after_binding=True,
            quality_scope='Actual302/492-row all40-layer consumed stages exactly match parent. Other shapes require serving confirmation.',
            full_optimizer_qualified=False),indent=2)+'\n')
        return value
