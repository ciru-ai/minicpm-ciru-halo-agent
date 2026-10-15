"""Current92-pin model with canonical reducer key and actual-shape preparation."""
import hashlib
import json
from pathlib import Path

from model60_metadata_partition_worker import Worker as Parent


class Worker(Parent):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        import attention_w8
        from attention_specializations import canonical_forward, Templates
        attention_w8.forward = canonical_forward()
        self._attention_templates = Templates()
        return result

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        import attention_specializations as helper
        templates = self._attention_templates
        try:
            receipt = templates.prepare()
        finally:
            templates.close()
            del self._attention_templates
        receipt.update(worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            helper_sha256=hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest(),
            delta='Unused ROWS constexpr is0; stage1 MAX_Q1..16 and reducer REQS1..8 prepared before ready',
            current_parent=Parent.__module__, full_optimizer_qualified=False)
        (self.run/'attention-specialization-warmup.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return result
