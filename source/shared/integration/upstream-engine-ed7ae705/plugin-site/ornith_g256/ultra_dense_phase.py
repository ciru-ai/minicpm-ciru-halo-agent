"""CPU request phases for selected dense projections; no device metadata reads."""
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps


@dataclass(frozen=True)
class Plan:
    requests: tuple
    segments: tuple  # (first, end, is_prefill)
    rows: int


def plan_from_requests(request_ids, counts, computed, prompts):
    if not (len(request_ids) == len(counts) == len(computed) == len(prompts)):
        raise ValueError('Dense phase metadata lengths differ')
    if not request_ids or len(set(request_ids)) != len(request_ids):
        raise ValueError('Dense phases require unique nonempty request IDs')
    segments, requests, offset = [], [], 0

    def append(first, end, prefill):
        if end == first:
            return
        if segments and segments[-1][2] == prefill:
            segments[-1] = (segments[-1][0], end, prefill)
        else:
            segments.append((first, end, prefill))

    for req, count, done, prompt in zip(request_ids, counts, computed, prompts):
        count, done, prompt = int(count), int(done), int(prompt)
        if count <= 0 or done < 0 or prompt <= 0:
            raise ValueError('Invalid dense phase token counts')
        boundary = offset + min(count, max(prompt - done, 0))
        append(offset, boundary, True)
        append(boundary, offset + count, False)
        requests.append((req, count, done, prompt))
        offset += count
    return Plan(tuple(requests), tuple(segments), offset)


_STATE = ('synthetic', None)
_INSTALLED = False
_LIVE_ROWS = None


@contextmanager
def scope(kind, value=None):
    global _STATE
    if kind not in ('synthetic', 'pending', 'real', 'capture'):
        raise ValueError('Unknown dense phase scope')
    if kind == 'real' and not isinstance(value, Plan):
        raise ValueError('Real dense phase requires an admitted CPU plan')
    if kind == 'capture' and value not in (1, 8, 16, 32, 48, 64):
        raise ValueError('Unsupported uniform decode capture')
    previous, _STATE = _STATE, (kind, value)
    try:
        yield
    finally:
        _STATE = previous


def segments_for(rows):
    kind, value = _STATE
    if rows <= 0:
        raise ValueError('Empty dense projection')
    if kind == 'pending':
        raise RuntimeError('Real dense call has no prepared CPU phase plan')
    if kind == 'capture':
        if rows != value:
            raise ValueError('Dense capture rows differ from uniform descriptor')
        return ((0, rows, False),)
    if kind == 'synthetic':
        return ((0, rows, True),)
    if value.rows > rows:
        raise ValueError('Dense tensor truncates admitted CPU phase rows')
    result = value.segments
    if value.rows < rows:
        result += ((value.rows, rows, True),)  # Eager padding stays BF16.
    return result


def bind_live_rows(tensor):
    """Bind once before capture; this scalar belongs only to the target model."""
    global _LIVE_ROWS
    import torch
    if (tensor.dtype != torch.int32 or tuple(tensor.shape) != (1,)
            or tensor.device.type != 'cuda' or not tensor.is_contiguous()):
        raise ValueError('Dense liveness requires a persistent device int32 scalar')
    if _LIVE_ROWS is not None:
        raise RuntimeError('Dense live-row scalar may only be bound once')
    _LIVE_ROWS = tensor


def update_live_rows(rows):
    """One current-stream fill before target work, never recorded in its graph."""
    import torch
    if _LIVE_ROWS is None or not 1 <= rows <= 8192:
        raise RuntimeError('Dense live-row update lacks a bound scalar or valid count')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Dense live-row update is forbidden inside capture')
    _LIVE_ROWS.fill_(rows)


def capture_live_pointer(tensor):
    if _LIVE_ROWS is None or tensor.data_ptr() != _LIVE_ROWS.data_ptr():
        raise RuntimeError('Dense live-row tensor differs from the bound target scalar')
    # Eager native slices already have exact M, including mixed row offsets.
    # Only a captured pure-decode graph needs a changing live prefix count.
    return tensor.data_ptr() if _STATE[0] == 'capture' else None


def make_determine_wrapper(original):
    """Keep parent padded FULL eligibility and require a current semantic plan."""
    @wraps(original)
    def determine(runner, num_tokens, num_reqs, num_scheduled_tokens_np,
                  max_num_scheduled_tokens, use_cascade_attn, allow_microbatching=True,
                  force_eager=False, force_uniform_decode=None, force_has_lora=None,
                  force_num_active_loras=None, num_encoder_reqs=0):
        result = original(runner, num_tokens, num_reqs, num_scheduled_tokens_np,
            max_num_scheduled_tokens, use_cascade_attn,
            allow_microbatching=allow_microbatching, force_eager=force_eager,
            force_uniform_decode=force_uniform_decode, force_has_lora=force_has_lora,
            force_num_active_loras=force_num_active_loras, num_encoder_reqs=num_encoder_reqs)
        if force_uniform_decode is None and num_tokens > 0:
            kind, plan = _STATE
            if kind != 'real' or plan.rows != num_tokens:
                raise RuntimeError('Real graph dispatch has no current dense phase plan')
            if getattr(result[0], 'name', None) == 'FULL':
                if any(prefill for _, _, prefill in plan.segments):
                    raise RuntimeError('Existing semantic graph guard admitted prompt rows as FULL decode')
                if result[1].num_tokens < num_tokens:
                    raise RuntimeError('FULL graph truncates real token rows')
        return result
    return determine


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    prepare_original = GPUModelRunner._prepare_inputs
    execute_original = GPUModelRunner.execute_model
    capture_original = GPUModelRunner._warmup_and_capture

    @wraps(prepare_original)
    def prepare(runner, scheduler_output, num_scheduled_tokens):
        global _STATE
        if runner.use_async_scheduling:
            raise ValueError('Dense phase plan requires synchronous scheduling')
        b, n = runner.input_batch, runner.input_batch.num_reqs
        plan = plan_from_requests(tuple(b.req_ids[:n]), tuple(num_scheduled_tokens),
                                  tuple(b.num_computed_tokens_cpu[:n]), tuple(b.num_prompt_tokens[:n]))
        if sum(scheduler_output.num_scheduled_tokens.values()) != plan.rows:
            raise ValueError('Dense phase plan differs from scheduler total')
        result = prepare_original(runner, scheduler_output, num_scheduled_tokens)
        update_live_rows(plan.rows)
        _STATE = ('real', plan)
        return result

    @wraps(execute_original)
    def execute(runner, *args, **kwargs):
        with scope('pending'):
            return execute_original(runner, *args, **kwargs)

    @wraps(capture_original)
    def capture(runner, desc, cudagraph_runtime_mode, *args, **kwargs):
        uniform_full = getattr(cudagraph_runtime_mode, 'name', None) == 'FULL' and desc.uniform
        if uniform_full:
            update_live_rows(desc.num_tokens)
        with scope('capture', desc.num_tokens) if uniform_full else scope('synthetic'):
            return capture_original(runner, desc, cudagraph_runtime_mode, *args, **kwargs)

    GPUModelRunner._prepare_inputs = prepare
    GPUModelRunner.execute_model = execute
    GPUModelRunner._warmup_and_capture = capture
    GPUModelRunner._determine_batch_execution_and_padding = make_determine_wrapper(
        GPUModelRunner._determine_batch_execution_and_padding)
    _INSTALLED = True
