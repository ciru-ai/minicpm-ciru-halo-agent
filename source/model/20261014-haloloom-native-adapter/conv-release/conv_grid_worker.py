"""Retained model60 plus a grid permutation of its corrected convolution.

Only the non-APC, non-speculative BF16 prefill specialization is eligible.
The native wrapper still prepares all metadata and owns its state buffers.
"""
import collections
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
from attention_w8_worker import Worker as Parent

HERE = Path(__file__).resolve().parent
RETAINED_SHA = 'e5b03321c3fb94240f915973a5008b62b7e765dfcf3a8d5805e2bbd6f1f6d5d8'


def eligible(b):
    if (b['IS_APC_ENABLED'] or b['IS_SPEC_DECODING'] or b['launch_pdl'] or
            (b['dim'], b['KERNEL_WIDTH'], b['BLOCK_M'], b['BLOCK_N']) != (8192, 4, 8, 256)):
        return False
    x, out, state = (b[n] for n in ('x_ptr', 'o_ptr', 'initial_states_ptr'))
    if not all(t.dtype == torch.bfloat16 for t in (x, out, state, b['w_ptr'])):
        return False
    if len({t.untyped_storage().data_ptr() for t in (x, out, state)}) != 3:
        return False
    return (b['stride_x_dim'] == b['stride_o_dim'] == b['stride_istate_dim'] == 1 and
            min(b['stride_x_token'], b['stride_o_token'], b['stride_istate_token']) >= b['dim'])


def ownership(b, grid):
    """Audit actual live metadata; no CPU readback is added to clean serving."""
    starts = b['query_start_loc_ptr'].cpu().tolist()
    batches = b['batch_ptr'][:grid[0]].cpu().tolist()
    offsets = b['token_chunk_offset_ptr'][:grid[0]].cpu().tolist()
    cache = b['cache_indices_ptr'].cpu()
    sequences = {}
    for seq, chunk in zip(batches, offsets):
        if seq == b['pad_slot_id']:
            continue
        slot = int(cache[seq].flatten()[0])
        if b['HAS_NULL_BLOCK'] and slot == b['null_block_id']:
            continue
        assert 0 <= slot < b['num_cache_lines']
        sequences.setdefault(seq, dict(slot=slot, chunks=[], tokens=starts[seq+1]-starts[seq]))['chunks'].append(chunk)
    slots = [s['slot'] for s in sequences.values()]
    assert len(slots) == len(set(slots)), 'Live sequences share a mutable state slot'
    for s in sequences.values():
        assert sorted(s['chunks']) == list(range((s['tokens'] + 7)//8))
    return dict(sequences=len(sequences), tokens=sum(s['tokens'] for s in sequences.values()),
                distinct_mutable_slots=True, complete_chunk_coverage=True)


class Worker(Parent):
    def load_model(self, *args, **kwargs):
        value = super().load_model(*args, **kwargs)
        from vllm.model_executor.layers.mamba.ops import causal_conv1d as conv
        from conv_grid_swap import _causal_conv1d_fwd_kernel_grid_swap as child
        source = Path(inspect.getfile(conv))
        assert hashlib.sha256(source.read_bytes()).hexdigest() == RETAINED_SHA
        pins = json.loads((HERE/'conv-prefill/candidate-pins.json').read_text())
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in pins.items())
        self._cv_ready = False
        self._cv_audit = os.environ.get('CIRU_CONV_GRID_AUDIT') == '1'
        self._cv_stats = torch.zeros(5, dtype=torch.int64, device=self.device) if self._cv_audit else None
        self._cv_context = {'cohort': 'startup'}
        self._cv_binding = dict(pins=pins, audit=self._cv_audit, kernels={}, shapes={},
            eligible_calls=0, fallback_calls=0, audited_ownership_calls=0,
            parent_source=dict(path=str(source), sha256=RETAINED_SHA),
            delta='Only transpose the two launch grid axes and program_id uses; unchanged corrected arithmetic and state updates',
            scope='BF16 dim8192 width4, native non-APC/non-speculative convolution; all other routes retain original',
            full_optimizer_qualified=False)
        self._cv_dirty = True
        original = conv._causal_conv1d_fwd_kernel.run
        names = list(conv._causal_conv1d_fwd_kernel.arg_names)
        assert names == list(child.arg_names)

        def launch(*a, **kw):
            b = dict(zip(names, a))
            b.update({n: kw[n] for n in names if n in kw})
            if not eligible(b):
                self._cv_binding['fallback_calls'] += 1
                return original(*a, **kw)
            auditing = self._cv_audit and self._cv_ready
            saved = {}
            native_grid = kw['grid']

            def grid(meta):
                dimensions = tuple(native_grid(meta) if callable(native_grid) else native_grid)
                saved.update(bound={n: meta[n] for n in names}, grid=dimensions)
                if auditing:
                    assert not torch.cuda.is_current_stream_capturing()
                    saved['ownership'] = ownership(saved['bound'], dimensions)
                    saved['state_before'] = meta['initial_states_ptr'].clone()
                    saved['out_before'] = meta['o_ptr'].clone()
                return (dimensions[1], dimensions[0])

            kernel = child.run(*a, **dict(kw, grid=grid))
            bound = saved['bound']
            if auditing:
                state, out = bound['initial_states_ptr'], bound['o_ptr']
                child_state, child_out = state.clone(), out.clone()
                # Replay the retained kernel on the ORIGINAL strided allocations.
                # Compact clones are snapshots only, never kernel arguments.
                state.copy_(saved['state_before'])
                out.copy_(saved['out_before'])
                launch_options = {k: v for k, v in kw.items() if k not in names and k != 'grid'}
                original(*[bound[n] for n in names], grid=saved['grid'], **launch_options)
                self._cv_stats[0].add_(1)
                self._cv_stats[1].add_(out.numel())
                self._cv_stats[2].add_(state.numel())
                self._cv_stats[3].add_((out.view(torch.int16) != child_out.view(torch.int16)).sum())
                self._cv_stats[4].add_((state.view(torch.int16) != child_state.view(torch.int16)).sum())
                state.copy_(child_state)
                out.copy_(child_out)
                self._cv_binding['audited_ownership_calls'] += 1
            self._cv_binding['eligible_calls'] += 1
            digest = hashlib.sha256(kernel.asm['hsaco']).hexdigest()
            if digest not in self._cv_binding['kernels']:
                self._cv_binding['kernels'][digest] = dict(function=kernel.function,
                    compiler_hash=kernel.hash, registers=kernel.n_regs, spills=kernel.n_spills,
                    num_warps=kernel.metadata.num_warps, shared=kernel.metadata.shared)
                folder = self.run/'conv-grid-binaries'/digest
                folder.mkdir(parents=True, exist_ok=True)
                for ext in ('hsaco', 'amdgcn', 'ttgir'):
                    raw = kernel.asm[ext]
                    (folder/('conv.'+ext)).write_bytes(raw if isinstance(raw, bytes) else raw.encode())
                self._cv_dirty = True
            key = json.dumps([self._cv_context['cohort'], list(bound['x_ptr'].shape),
                              list(bound['x_ptr'].stride()), list(bound['o_ptr'].stride()),
                              list(bound['initial_states_ptr'].stride()), saved['grid'], digest])
            if key not in self._cv_binding['shapes']:
                self._cv_binding['shapes'][key] = dict(context=dict(self._cv_context),
                    input_shape=list(bound['x_ptr'].shape), input_stride=list(bound['x_ptr'].stride()),
                    output_stride=list(bound['o_ptr'].stride()), state_stride=list(bound['initial_states_ptr'].stride()),
                    parent_grid=list(saved['grid']), candidate_grid=list(reversed(saved['grid'])),
                    binary_sha256=digest, calls=0, audit=auditing)
                self._cv_dirty = True
            self._cv_binding['shapes'][key]['calls'] += 1
            return kernel

        conv._causal_conv1d_fwd_kernel.run = launch
        self._save_conv_grid()
        return value

    def _save_conv_grid(self):
        if self._cv_audit and self._cv_ready:
            values = self._cv_stats.cpu().tolist()
            audit = dict(zip(('calls', 'output_values', 'state_values', 'output_mismatches', 'state_mismatches'), values))
            audit.update(scope='All eligible live prefill calls after warmup; identical operands and original strided state/output allocations',
                         serves='candidate', diagnostic_only=True)
            (self.run/'conv-grid-live-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        (self.run/'conv-grid-binding.json').write_text(json.dumps(self._cv_binding, indent=2)+'\n')
        self._cv_dirty = False

    def compile_or_warm_up_model(self):
        value = super().compile_or_warm_up_model()
        self._cv_ready = True
        self._save_conv_grid()
        return value

    def execute_model(self, scheduler_output, *args, **kwargs):
        if not getattr(self, '_cv_ready', False):
            return super().execute_model(scheduler_output, *args, **kwargs)
        if scheduler_output.scheduled_new_reqs:
            control = os.environ.get('CIRU_DEPTH_CONTROL')
            self._cv_context['cohort'] = json.loads(Path(control).read_text())['label'] if control else 'serving'
        counts = scheduler_output.num_scheduled_tokens
        self._cv_context.update(actual_requests=len(counts), scheduled_tokens=sum(counts.values()),
            drafts={k: len(v) for k, v in scheduler_output.scheduled_spec_decode_tokens.items()})
        before = self._cv_binding['eligible_calls']
        value = super().execute_model(scheduler_output, *args, **kwargs)
        if self._cv_binding['eligible_calls'] != before:
            self._save_conv_grid()
        return value
