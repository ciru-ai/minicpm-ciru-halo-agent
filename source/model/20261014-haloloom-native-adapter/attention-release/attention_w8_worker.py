"""Preserved model60 plus the one verified attention warp-distribution child."""
import hashlib
import json
import os
from pathlib import Path

import torch
from kernel_identity import loaded_identity
from gu64k_worker import Worker as Parent

HERE = Path(__file__).resolve().parent


class Worker(Parent):
    def load_model(self, *args, **kwargs):
        value = super().load_model(*args, **kwargs)
        import attention_shared_query as shared
        import attention_w8 as child
        pins = json.loads((HERE / 'attention-register/candidate-pins.json').read_text())
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in pins.items())
        self._w8_audit = os.environ.get('CIRU_ATTENTION_W8_AUDIT') == '1'
        self._w8_ready = False
        self._w8_stats = torch.zeros(5, device=self.device) if self._w8_audit else None
        self._w8_binding = dict(pins=pins, audit=self._w8_audit, kernels={},
            delta='Stage1 num_warps four to eight; unchanged BF16 math, split count, reducer and KV layout',
            parent='Registered model60 GU producer plus retained GDN/BF16/adaptive stack',
            full_optimizer_qualified=False)
        original_run = shared.shared_query_stage1.run
        parent_forward = shared.forward

        def launch(*a, **kw):
            kernel = original_run(*a, **kw)
            digest = hashlib.sha256(kernel.asm['hsaco']).hexdigest()
            if digest not in self._w8_binding['kernels']:
                self._w8_binding['kernels'][digest] = dict(
                    compiler_hash=kernel.hash, function=kernel.function,
                    num_warps=kernel.metadata.num_warps, registers=kernel.n_regs,
                    reported_spills=kernel.n_spills, shared=kernel.metadata.shared,
                    max_query=kw['MAX_Q'], query_stride=[kw['QS'], kw['QH']],
                    grid=list(kw['grid']), capture=torch.cuda.is_current_stream_capturing())
                if kernel.metadata.num_warps == 8 and kw['MAX_Q'] in (3, 4):
                    assert digest in {
                        'b0620fb69c958a31c053036282ed794af39b824e5a409b825ebf10e89e4427b0',
                        '7f7b33132e4686ceaf2817f433a86ab6e943464ed7f01ce548eb3ad6dac398f6'} or loaded_identity(kernel.asm['hsaco']) in {
                        '4a2279c6071bcbd46123de8a1de93191d111345090e9beede17a867a3a4d4112',
                        '5d39c4b48c793d79744822b00a0d233aa34e0df9c6aaefcd5c676f3a44ec414a'}, digest
                self._save_w8()
            return kernel

        shared.shared_query_stage1.run = launch

        def forward(q, k, v, out, table, starts, seq, scale, ks, vs, **options):
            result = child.forward(q, k, v, out, table, starts, seq, scale, ks, vs, **options)
            if self._w8_stats is not None:
                reference = torch.empty_like(out)
                parent_forward(q, k, v, reference, table, starts, seq, scale, ks, vs, **options)
                live = torch.arange(q.shape[0], device=q.device) < starts[-1]
                active = live[:, None, None]
                diff = torch.where(active, out.float()-reference.float(), 0.)
                self._w8_stats[0].add_(1)
                self._w8_stats[1].add_(live.sum()*4096)
                self._w8_stats[2].copy_(torch.maximum(self._w8_stats[2], diff.abs().nan_to_num().amax()))
                self._w8_stats[3].add_((active & (out != reference)).sum())
                self._w8_stats[4].add_((active & (~out.isfinite() | ~reference.isfinite())).sum())
            return result

        shared.forward = forward
        self._save_w8()
        return value

    def _save_w8(self):
        (self.run / 'attention-w8-binding.json').write_text(json.dumps(self._w8_binding, indent=2) + '\n')
        if self._w8_ready and self._w8_stats is not None:
            values = self._w8_stats.cpu().tolist()
            record = dict(zip(('calls', 'compared_values', 'max_abs', 'mismatches', 'nonfinite'), values))
            record.update(scope='Candidate and retained attention on identical live operands, including graph replay; warmup reset',
                          serves='candidate', diagnostic_only=True)
            (self.run / 'attention-w8-live-audit.json').write_text(json.dumps(record, indent=2) + '\n')

    def compile_or_warm_up_model(self):
        value = super().compile_or_warm_up_model()
        if self._w8_stats is not None:
            self._w8_stats.zero_()
        self._w8_ready = True
        self._save_w8()
        return value

    def _save_attention(self):
        super()._save_attention()
        if getattr(self, '_w8_ready', False) and self._w8_stats is not None:
            self._save_w8()
