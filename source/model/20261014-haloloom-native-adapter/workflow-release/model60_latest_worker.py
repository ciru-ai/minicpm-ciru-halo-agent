"""Model60 consolidation: retained draft heads plus the exact C5 target graph."""
from dataclasses import replace
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys

# This uniquely named entrypoint loads the frozen child copies beside itself.
# Their original parent still resolves to the retained draft-head release.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from draft_head_small_worker import Worker as Parent


class Worker(Parent):
    def compile_or_warm_up_model(self):
        from vllm.config import CUDAGraphMode
        manager = self.model_runner.cudagraph_manager
        assert manager is not None and not manager._graphs_captured
        descs = manager._capture_descs[CUDAGraphMode.FULL]
        template = next(d for d in descs if d.num_tokens == 18 and d.num_reqs == 6
                        and d.uniform_token_count == 3 and d.num_active_loras == 0)
        exact = replace(template, num_tokens=15, num_reqs=5)
        assert exact not in descs
        parent_candidates = {key: tuple(value) for key, value in manager._candidates.items()}
        descs.append(exact)
        descs.sort(key=lambda d: d.num_tokens, reverse=True)
        result = super().compile_or_warm_up_model()
        assert exact in manager.graphs
        assert parent_candidates == {key: tuple(value) for key, value in manager._candidates.items()}
        cases = [(r, r*q, q, 0, q) for r in range(1, 9) for q in (1, 2, 3, 4)]
        before = {case: manager.dispatch(*case) for case in cases}
        # Append to one native lookup bucket. Existing compatible choices keep
        # their priority; the measured C5 hole alone receives the exact graph.
        key = (15, 0)
        manager._candidates[key] = [*manager._candidates.get(key, []), exact]
        after = {case: manager.dispatch(*case) for case in cases}
        changed = [case for case in cases if before[case] != after[case]]
        assert changed == [(5, 15, 3, 0, 3)], changed
        assert after[changed[0]] == exact
        assert before[changed[0]].cg_mode == CUDAGraphMode.NONE
        assert all(tuple(value) == parent_candidates[k] for k, value in manager._candidates.items() if k != key)
        binding = dict(worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            worker_file=str(Path(__file__)), head_worker_file=inspect.getfile(Parent),
            exact_target_graph=str(exact), changed_native_dispatch_cases=changed,
            all_other_checked_shapes_unchanged=True, target_model_arithmetic_changed=False,
            retained_c1_and_c8_heads=True, small_heads=['M4/C2','M6/C3'],
            runtime_source_modified=False, diagnostic_sampling_audit=False,
            known_risk='One prior small-child C3 vision repeat had an unexplained0.833s decode stall; retained in performance evidence.',
            full_optimizer_qualified=False)
        (self.run/'model60-latest-binding.json').write_text(json.dumps(binding, indent=2)+'\n')
        original = manager.run_fullgraph
        observed = set()

        @functools.wraps(original)
        def run_graph(desc, *args, **kwargs):
            value = original(desc, *args, **kwargs)
            shape = (desc.num_tokens, desc.num_reqs, desc.uniform_token_count)
            if shape not in observed:
                observed.add(shape)
                with (self.run/'model60-target-graph-dispatch.jsonl').open('a') as file:
                    file.write(json.dumps(dict(shape=shape, exact_c5=desc==exact))+'\n')
            return value

        manager.run_fullgraph = run_graph
        return result
