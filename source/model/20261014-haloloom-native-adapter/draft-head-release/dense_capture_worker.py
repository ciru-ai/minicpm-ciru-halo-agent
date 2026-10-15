"""Observe retained GEMMs inside real HIP graph replay; no math replacement.

Named GPU bookends bind each native GEMM to its module and physical operands.
In-graph snapshots precede allocator reuse. Only representative weight banks
are written, once; captured activations cover early and late C1/C8 steps.
"""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from conv_profile_worker import Worker as Parent


@triton.jit
def dense_observe_before(markers, ordinal):
    tl.store(markers + ordinal * 2, ordinal + 1)


@triton.jit
def dense_observe_after(markers, ordinal):
    tl.store(markers + ordinal * 2 + 1, ordinal + 1)


def describe(t):
    if t is None:
        return None
    return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                storage_offset=t.storage_offset(), storage_bytes=t.untyped_storage().nbytes(),
                storage_address=t.untyped_storage().data_ptr(), address=t.data_ptr())


class Worker(Parent):
    def load_model(self, *args, **kwargs):
        value = super().load_model(*args, **kwargs)
        self._dc_context = None
        self._dc_graphs = {}
        self._dc_replay = None
        self._dc_done = set()
        self._dc_banks = []
        self._dc_weights = {}
        self._dc_ready = False
        self._dc_control = {'label': 'startup'}
        self._dc_out = self.run/'dense-operands'
        self._dc_out.mkdir()
        self._dc_owners = {}
        candidates = {}
        for owner, model in (('target', self.model_runner.get_model()),
                              ('draft', self.model_runner.speculator.model)):
            for name, module in model.named_modules():
                w = getattr(module, 'weight', None)
                if not isinstance(w, torch.Tensor) or w.ndim != 2:
                    continue
                row = dict(owner=owner, name=name, weight=describe(w),
                           method=type(getattr(module, 'quant_method', None)).__name__)
                self._dc_owners.setdefault(w.untyped_storage().data_ptr(), []).append(row)
                if owner == 'target' and name.startswith('language_model.') and tuple(w.shape) in (
                        (12288, 2048), (9216, 2048), (2048, 4096)):
                    candidates.setdefault(tuple(w.shape), []).append((name, w))
        self._dc_selected = {}
        for shape, modules in candidates.items():
            for name, w in (modules[0], modules[-1]):
                self._dc_selected[w.data_ptr()] = name
        assert len(self._dc_selected) == 6, self._dc_selected
        self._dc_original_linear = torch.nn.functional.linear

        @functools.wraps(self._dc_original_linear)
        def linear(x, weight, bias=None):
            # The ROCm custom op calls F.linear with real tensors, including
            # during graph construction. Dynamo's unrelated traces stay native.
            if torch.compiler.is_compiling() or self._dc_context is None:
                return self._dc_original_linear(x, weight, bias)
            if not torch.cuda.is_current_stream_capturing():
                return self._dc_original_linear(x, weight, bias)
            context = self._dc_context
            ordinal = len(context['calls'])
            assert ordinal < 512
            owners = self._dc_owners.get(weight.untyped_storage().data_ptr(), [])
            call = dict(ordinal=ordinal, api='torch.nn.functional.linear',
                        input=describe(x), weight=describe(weight), bias=describe(bias),
                        owners=owners, graph_capture=True)
            context['calls'].append(call)
            selected = weight.data_ptr() in self._dc_selected
            snap = None
            if selected:
                snap = dict(input=x.clone(), weight_live=weight, bias_live=bias)
            dense_observe_before[(1,)](context['markers'], ordinal, num_warps=1)
            result = self._dc_original_linear(x, weight, bias)
            dense_observe_after[(1,)](context['markers'], ordinal, num_warps=1)
            call['output'] = describe(result)
            if selected:
                snap.update(expected=result.clone(), call=call,
                            module=self._dc_selected[weight.data_ptr()])
                context['snapshots'].append(snap)
            return result

        torch.nn.functional.linear = linear
        return value

    def compile_or_warm_up_model(self):
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
        marker_warmup = torch.empty(2, dtype=torch.int32, device=self.device)
        dense_observe_before[(1,)](marker_warmup, 0, num_warps=1)
        dense_observe_after[(1,)](marker_warmup, 0, num_warps=1)
        manager = self.model_runner.cudagraph_manager
        original_capture = CudaGraphManager.capture

        def capture(obj, create_forward_fn, *args, **kwargs):
            def create(desc, *a, **kw):
                forward = create_forward_fn(desc, *a, **kw)
                key = (desc.num_tokens, desc.num_reqs, desc.uniform_token_count)
                selected = obj is manager and key in ((4, 1, 4), (24, 8, 3)) and not kw.get('warmup', True)
                if not selected:
                    return forward
                assert key not in self._dc_graphs, key
                context = dict(key=key, calls=[], snapshots=[],
                               markers=torch.empty(1024, dtype=torch.int32, device=self.device))
                self._dc_graphs[key] = context

                def observed(*fa, **fk):
                    assert self._dc_context is None
                    self._dc_context = context
                    try:
                        return forward(*fa, **fk)
                    finally:
                        self._dc_context = None
                return observed
            return original_capture(obj, create, *args, **kwargs)

        CudaGraphManager.capture = capture
        try:
            value = super().compile_or_warm_up_model()
        finally:
            CudaGraphManager.capture = original_capture
        assert set(self._dc_graphs) == {(4, 1, 4), (24, 8, 3)}, self._dc_graphs.keys()
        for key, context in self._dc_graphs.items():
            assert len(context['calls']) >= 270, (key, len(context['calls']))
            assert len(context['snapshots']) == 6, (key, len(context['snapshots']))
        original_graph = manager.run_fullgraph

        def replay(desc, *args, **kwargs):
            self._dc_replay = (desc.num_tokens, desc.num_reqs, desc.uniform_token_count)
            return original_graph(desc, *args, **kwargs)
        manager.run_fullgraph = replay
        libraries = sorted({s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines()
                            if any(n in s.lower() for n in ('hipblas', 'rocblas', 'tensile')) and '/' in s})
        import vllm.model_executor.layers.utils as linear_utils
        paths = [Path(__file__), Path(linear_utils.__file__), Path(torch._C.__file__)]
        paths += [Path(p) for p in libraries if Path(p).is_file()]
        self._dc_binding = dict(torch=torch.__version__, hip=torch.version.hip,
            blas_preference=str(torch.backends.cuda.preferred_blas_library()),
            loaded_sources_libraries={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        self._dc_ready = True
        self._save_dense_index()
        return value

    def _save_dense_index(self):
        graphs = [dict(key=list(k), calls=v['calls'], snapshots=len(v['snapshots']))
                  for k, v in self._dc_graphs.items()]
        (self.run/'dense-capture-index.json').write_text(json.dumps(dict(
            graphs=graphs, banks=self._dc_banks, weights=self._dc_weights,
            completed=sorted(self._dc_done), binding=self._dc_binding,
            math_changed=False, diagnostic_only=True, performance_eligible=False,
            full_optimizer_qualified=False,
            scope='GPU bookends around retained F.linear; graph snapshots refreshed by actual selected target replay'),
            indent=2)+'\n')

    def execute_model(self, scheduler_output, *args, **kwargs):
        if not self._dc_ready:
            return super().execute_model(scheduler_output, *args, **kwargs)
        if scheduler_output.scheduled_new_reqs:
            self._dc_control = json.loads(Path(os.environ['CIRU_DEPTH_CONTROL']).read_text())
        counts = scheduler_output.num_scheduled_tokens
        drafts = scheduler_output.scheduled_spec_decode_tokens
        c = len(counts)
        outputs = list(map(int, scheduler_output.scheduled_cached_reqs.num_output_tokens))
        shape = None
        if c in (1, 8) and outputs and all(n == (4 if c == 1 else 3)
                and len(drafts.get(k, [])) == n-1 for k, n in counts.items()):
            shape = 'late' if min(outputs) >= 128 else 'early' if min(outputs) >= 8 else None
        label = self._dc_control['label']
        key = label+'/'+str(shape)
        selected = shape is not None and key not in self._dc_done
        self._dc_replay = None
        if selected:
            with torch.inference_mode():
                for graph in self._dc_graphs.values():
                    graph['markers'].zero_()
        value = super().execute_model(scheduler_output, *args, **kwargs)
        if selected:
            torch.cuda.synchronize(self.device)
            graph = self._dc_graphs[self._dc_replay]
            markers = graph['markers'].cpu().tolist()
            expected = [i+1 for i in range(len(graph['calls'])) for _ in range(2)]
            assert markers[:len(expected)] == expected, 'Graph bookends were not replayed completely'
            step = dict(cohort=label, shape=shape, graph=list(self._dc_replay),
                        scheduled_counts=dict(counts), drafts={k:len(v) for k,v in drafts.items()},
                        output_counts=outputs, observed_gemm_calls=len(graph['calls']), marker_pairs_valid=True)
            for snap in graph['snapshots']:
                module = snap['module']
                weight_file = module+'.weight.pt'
                if module not in self._dc_weights:
                    payload = dict(weight=snap['weight_live'].cpu(),
                                   bias=snap['bias_live'].cpu() if snap['bias_live'] is not None else None,
                                   original=snap['call']['weight'], module=module)
                    torch.save(payload, self._dc_out/weight_file)
                    self._dc_weights[module] = dict(file=weight_file,
                        sha256=hashlib.sha256((self._dc_out/weight_file).read_bytes()).hexdigest())
                name = label+'-'+shape+'-'+module+'.pt'
                payload = dict(input=snap['input'].cpu(), expected=snap['expected'].cpu(),
                    call=snap['call'], module=module, weight_file=weight_file, scheduler=step,
                    scope='Compact saved values; restore original shape/stride/offset for replay')
                torch.save(payload, self._dc_out/name)
                self._dc_banks.append(dict(file=name, module=module, scheduler=step))
            self._dc_done.add(key)
            self._save_dense_index()
        return value
