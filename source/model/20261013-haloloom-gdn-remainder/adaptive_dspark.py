"""Execute both retained DSpark query shapes against one model and KV cache."""
import json
from collections import Counter
from types import MethodType
import torch


def install(worker):
    p = worker.model_runner.speculator
    assert type(p).__name__ == 'DSparkSpeculator'
    assert p.num_speculative_steps == 3 and p.sample_from_anchor
    assert p.draft_logits is None and not p.enable_adaptive_verification
    assert p.vllm_config.parallel_config.tensor_parallel_size == 1
    original_init = p.init_cudagraph_manager
    original_capture = p.capture
    original_propose = p.propose
    managers = {}
    anchors = {k: torch.arange(p.max_num_reqs, device=p.device, dtype=torch.int64)*k for k in (2, 3)}
    columns = {k: torch.arange(k, device=p.device, dtype=torch.int32).repeat(p.max_num_reqs) for k in (2, 3)}
    p._depth_counts = Counter()
    p._depth_last = None
    last_policy = [None]
    seen = set()

    def select(k):
        assert k in (2, 3), k
        p.num_speculative_steps = k
        p.num_query_per_req = k
        p._anchor_idx = anchors[k]
        p.sample_col = columns[k]
        if managers:
            p.query_cudagraph_manager = managers[k]

    def init(self, mode):
        for k in (3, 2):
            # The maximum-sized buffers and shared model are never reallocated.
            self.num_speculative_steps = k
            self.num_query_per_req = k
            original_init(mode)
            managers[k] = self.query_cudagraph_manager
            original_run = managers[k].run_fullgraph

            def run(desc, _run=original_run, _k=k):
                key = (_k, desc.num_reqs, desc.num_tokens, desc.uniform_token_count)
                self._depth_counts[key] += 1
                self._depth_last = dict(depth=_k, requests=desc.num_reqs, tokens=desc.num_tokens,
                                        uniform=desc.uniform_token_count, graph=True)
                if key not in seen:
                    seen.add(key)
                    with (worker.run/'adaptive-draft-dispatch.jsonl').open('a') as f:
                        f.write(json.dumps(self._depth_last)+'\n')
                return _run(desc)
            managers[k].run_fullgraph = run
        select(3)

    def capture(self):
        for k in (3, 2):
            select(k)
            original_capture()
        select(3)
        (worker.run/'adaptive-graphs.json').write_text(json.dumps({
            'depths': [2, 3], 'shared_weight_parameters': sum(x.numel() for x in self.model.parameters()),
            'draft_buffer_shape': list(self.draft_tokens.shape),
            'graphs': {k: [str(d) for d in m.graphs] if hasattr(m, 'graphs') else str(vars(m).keys()) for k, m in managers.items()},
            'maximum_allocation_depth': 3, 'scheduler_schedule': [[1, 1, 3], [2, 8, 2]],
        }, indent=2)+'\n')

    def propose(self, input_batch, *args, **kwargs):
        k = kwargs.get('num_speculative_tokens')
        if k is None or kwargs.get('dummy_run', False) or not managers:
            k = 3
        select(int(k))
        self._depth_last = dict(depth=k, requests=input_batch.num_reqs,
                                tokens=input_batch.num_reqs*k, graph=False)
        # Returning the maximum-width buffer is required by runner request state.
        # The existing runner invalidates columns >= the scheduled depth.
        value = original_propose(input_batch, *args, **kwargs)
        assert value.shape == (input_batch.num_reqs, 3), value.shape
        if not kwargs.get('dummy_run', False) and last_policy[0] != k:
            with (worker.run/'adaptive-depth-transitions.jsonl').open('a') as f:
                f.write(json.dumps(dict(previous=last_policy[0], selected=k,
                    request_ids=input_batch.req_ids, actual_requests=input_batch.num_reqs,
                    target_tokens=input_batch.num_tokens, draft_dispatch=self._depth_last))+'\n')
            last_policy[0] = k
        return value

    p.init_cudagraph_manager = MethodType(init, p)
    p.capture = MethodType(capture, p)
    p.propose = MethodType(propose, p)

