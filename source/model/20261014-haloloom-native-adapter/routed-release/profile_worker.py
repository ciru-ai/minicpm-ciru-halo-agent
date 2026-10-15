"""Selected current model60 regions; diagnostic timings only, math unchanged."""
import collections
import ctypes as C
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch
from remainder_worker import Worker as Parent

SDK = Path('/srv/ssd/p3700ba/scratch/crown/rocprof-sdk-alignment-fix-20261013/install')


class Roctx:
    def __init__(self):
        self.lib = C.CDLL(str(SDK / 'lib/librocprofiler-sdk-roctx.so.1'))
        for name in ('roctxProfilerPause', 'roctxProfilerResume'):
            fn = getattr(self.lib, name)
            fn.argtypes = [C.c_uint64]
            fn.restype = C.c_int
        self.lib.roctxRangePushA.argtypes = [C.c_char_p]
        self.lib.roctxRangePushA.restype = C.c_int
        self.lib.roctxRangePop.argtypes = []
        self.lib.roctxRangePop.restype = C.c_int

    def pause(self):
        assert self.lib.roctxProfilerPause(0) == 0

    def resume(self):
        assert self.lib.roctxProfilerResume(0) == 0

    def push(self, label):
        result = self.lib.roctxRangePushA(label.encode())
        assert result >= 0
        return result

    def pop(self, depth):
        assert self.lib.roctxRangePop() == depth

    def detach(self):
        # Version-pinned private SDK: attach.cpp exports status_t(void).
        # Export while the spawned worker lives; multiprocessing shutdown does
        # not reliably execute this SDK's finalizer.
        assert os.environ.get('ROCPROF_ATTACH_OUTPUT_GENERATION_SYNC') == '1'
        sdk = C.CDLL(str(SDK / 'lib/librocprofiler-sdk.so.1'))
        sdk.rocprofiler_detach.argtypes = []
        sdk.rocprofiler_detach.restype = C.c_int
        assert sdk.rocprofiler_detach() == 0


class Worker(Parent):
    def compile_or_warm_up_model(self):
        value = super().compile_or_warm_up_model()
        self._profile_roctx = Roctx()
        self._profile_roctx.pause()
        self._profile_rows = []
        self._profile_pending = None
        self._profile_detached = False
        self._profile_skipped = collections.Counter()
        self._profile_histogram = collections.Counter()
        self._profile_control = {'label': 'startup', 'trace': False}
        r = self.model_runner
        for name in ('prepare_inputs', 'prepare_attn', 'sample', 'postprocess_sampled', 'postprocess_num_computed_tokens'):
            self._wrap(r, name)
        self._wrap(r.get_model(), 'compute_logits', 'target_head')
        if r.speculator is not None:
            self._wrap(r.speculator, 'propose', 'draft_propose')
        original_graph = r.cudagraph_manager.run_fullgraph

        def graph(desc, *args, **kwargs):
            if self._profile_pending is not None:
                self._profile_pending['target_graph'] = dict(requests=desc.num_reqs,
                    tokens=desc.num_tokens, uniform=desc.uniform_token_count, max_query=desc.max_query_len)
            return original_graph(desc, *args, **kwargs)

        r.cudagraph_manager.run_fullgraph = graph
        inventory = []
        for owner, model in (('target', r.get_model()), ('draft', r.speculator.model)):
            for name, module in model.named_modules():
                weight = getattr(module, 'weight', None)
                if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                    inventory.append(dict(owner=owner, name=name, shape=list(weight.shape),
                        stride=list(weight.stride()), dtype=str(weight.dtype),
                        bytes=weight.numel()*weight.element_size(),
                        method=type(getattr(module, 'quant_method', None)).__name__))
        (self.run / 'linear-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
        sources = [Path(__file__), Path(inspect.getfile(Parent)), Path(inspect.getfile(type(r)))]
        (self.run / 'profile-source-binding.json').write_text(json.dumps(dict(
            sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            sdk=str(SDK), diagnostic_only=True, model60_source_edits=False), indent=2) + '\n')
        self._save_profile()
        return value

    def _wrap(self, owner, name, label=None):
        if not hasattr(owner, name):
            return
        original = getattr(owner, name)

        @functools.wraps(original)
        def call(*args, **kwargs):
            row = self._profile_pending
            if row is None:
                return original(*args, **kwargs)
            marker = f'M60_{row["index"]}_{label or name}'
            depth = self._profile_roctx.push(marker)
            start = time.monotonic_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self._profile_roctx.pop(depth)
                row['host_stages'].append(dict(name=label or name, marker=marker,
                    duration_ns=time.monotonic_ns()-start))
                if label == 'draft_propose':
                    row['draft_dispatch'] = dict(getattr(owner, '_depth_last', {}))

        setattr(owner, name, call)

    def _save_profile(self):
        required = {
            'c1-64k/prefill': 1, 'c1-64k/c1-early': 2, 'c1-64k/c1-late': 2,
            'c8-mixed/mixed': 1, 'c8-mixed/c8-early': 2, 'c8-mixed/c8-late': 2,
        }
        seen = collections.Counter(row['key'] for row in self._profile_rows)
        if not self._profile_detached and all(seen[key] >= count for key, count in required.items()):
            assert self._profile_pending is None
            torch.cuda.synchronize(self.device)
            start = time.monotonic_ns()
            self._profile_roctx.detach()
            self._profile_detached = True
            files = sorted(str(p.relative_to(self.run)) for p in (self.run / 'rocprof').rglob('*') if p.is_file())
            (self.run / 'profile-detach.json').write_text(json.dumps(dict(
                worker_pid=os.getpid(), status='EXPORTED_BEFORE_WORKER_EXIT',
                required=required, seen=dict(seen), elapsed_ns=time.monotonic_ns()-start,
                files=files, gpu_dispatch_join_pending=True), indent=2) + '\n')
        (self.run / 'live-profile.json').write_text(json.dumps(dict(
            steps=self._profile_rows, histogram=dict(self._profile_histogram),
            detached=self._profile_detached,
            diagnostic_only=True, performance_eligible=False, quality_evidence=False,
            full_optimizer_qualified=False), indent=2) + '\n')

    def execute_model(self, scheduler_output, *args, **kwargs):
        if not hasattr(self, '_profile_roctx') or self._profile_detached:
            return super().execute_model(scheduler_output, *args, **kwargs)
        if scheduler_output.scheduled_new_reqs:
            self._profile_control = json.loads(Path(os.environ['CIRU_DEPTH_CONTROL']).read_text())
        label = self._profile_control['label']
        counts = scheduler_output.num_scheduled_tokens
        drafts = scheduler_output.scheduled_spec_decode_tokens
        total = sum(counts.values())
        concurrency = len(counts)
        cached = scheduler_output.scheduled_cached_reqs
        outputs = list(map(int, cached.num_output_tokens))
        shape = None
        if counts and max(counts.values()) > 4:
            shape = 'mixed' if any(drafts.values()) else 'prefill'
        elif concurrency in (1, 8) and all(n == (4 if concurrency == 1 else 3)
                and len(drafts.get(k, [])) == n - 1 for k, n in counts.items()):
            if outputs and min(outputs) >= 128:
                shape = f'c{concurrency}-late'
            elif outputs and min(outputs) >= 8:
                shape = f'c{concurrency}-early'
        key = label + '/' + str(shape)
        self._profile_histogram[key] += 1
        seen = sum(r['key'] == key for r in self._profile_rows)
        limit = 1 if shape in ('prefill', 'mixed') else 2
        if shape is None or seen >= limit or not self._profile_control.get('trace'):
            return super().execute_model(scheduler_output, *args, **kwargs)
        if shape == 'prefill' and self._profile_skipped[key] < 2:
            self._profile_skipped[key] += 1
            return super().execute_model(scheduler_output, *args, **kwargs)
        assert self._profile_pending is None
        row = dict(key=key, cohort=label, shape=shape, index=len(self._profile_rows),
            scheduled_tokens=total, requests=concurrency, host_stages=[], worker_pid=os.getpid(),
            scheduled_counts=dict(counts), draft_counts={k: len(v) for k, v in drafts.items()},
            cached_computed=list(map(int, cached.num_computed_tokens)), cached_output=outputs,
            next_depth=scheduler_output.num_spec_tokens_to_schedule)
        self._profile_pending = row
        marker = f'M60_{row["index"]}_TARGET'
        row['target_marker'] = marker
        torch.cuda.synchronize(self.device)
        self._profile_roctx.resume()
        depth = self._profile_roctx.push(marker)
        start = time.monotonic_ns()
        try:
            value = super().execute_model(scheduler_output, *args, **kwargs)
            torch.cuda.synchronize(self.device)
        finally:
            self._profile_roctx.pop(depth)
            self._profile_roctx.pause()
        row['target_wall_ms'] = (time.monotonic_ns()-start)/1e6
        if value is not None:
            self._profile_rows.append(row)
            self._profile_pending = None
            self._save_profile()
        return value

    def sample_tokens(self, *args, **kwargs):
        row = getattr(self, '_profile_pending', None)
        if row is None:
            return super().sample_tokens(*args, **kwargs)
        marker = f'M60_{row["index"]}_SAMPLE_DRAFT'
        row['sample_marker'] = marker
        torch.cuda.synchronize(self.device)
        self._profile_roctx.resume()
        depth = self._profile_roctx.push(marker)
        start = time.monotonic_ns()
        try:
            value = super().sample_tokens(*args, **kwargs)
            torch.cuda.synchronize(self.device)
        finally:
            self._profile_roctx.pop(depth)
            self._profile_roctx.pause()
        row['sample_wall_ms'] = (time.monotonic_ns()-start)/1e6
        self._profile_rows.append(row)
        self._profile_pending = None
        self._save_profile()
        return value
