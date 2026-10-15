"""Complete-Q8/head bindings, current stream, explicit shared-arena mutation."""
import ctypes as C
import hashlib
from pathlib import Path
import torch
from .ultra_contract import CompleteLayout, validate_settings
from . import native as parent_native

_Q8 = None
_Q8_PREFILL = None


def configure(settings, capacity):
    global _Q8, _Q8_PREFILL
    validate_settings(settings)
    path = str(Path(settings['q8_library']).resolve(strict=True))
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != settings['q8_library_sha256']:
        raise ValueError('Q8 library differs from the selected binary receipt')
    symbol = 'ornith_q8_complete_' + settings['q8_kernel'] + '_launch'
    identity = (path, digest, symbol)
    if _Q8 is not None and _Q8[0] != identity:
        raise RuntimeError('Cannot replace a Q8 kernel in a live worker')
    if _Q8 is None:
        lib = C.CDLL(path)
        query = lib.ornith_q8_complete_get_layout
        query.argtypes, query.restype = [C.c_int]*5 + [C.POINTER(CompleteLayout)], C.c_int
        launch = getattr(lib, symbol)
        launch.argtypes = [C.c_void_p]*10 + [C.c_size_t] + [C.c_void_p]*2 + [C.c_int]*6 + [C.c_void_p]
        launch.restype = C.c_int
        _Q8 = (identity, lib, query, launch)
    layout = CompleteLayout()
    parent_native.check(_Q8[2](capacity, 256, 2048, 512, 8, C.byref(layout)), 'Q8 layout')

    prefill_path = str(Path(settings['q8_prefill_library']).resolve(strict=True))
    prefill_digest = hashlib.sha256(Path(prefill_path).read_bytes()).hexdigest()
    if prefill_digest != settings['q8_prefill_library_sha256']:
        raise ValueError('Q8 prefill library differs from the selected binary receipt')
    prefill_symbol = settings['q8_prefill_symbol']
    prefill_identity = (prefill_path, prefill_digest, prefill_symbol, settings['q8_prefill_min_rows'])
    if _Q8_PREFILL is not None and _Q8_PREFILL[0] != prefill_identity:
        raise RuntimeError('Cannot replace Q8 prefill dispatch in a live worker')
    if _Q8_PREFILL is None:
        lib = C.CDLL(prefill_path)
        query = lib.ornith_q8_complete_get_layout
        query.argtypes, query.restype = [C.c_int]*5 + [C.POINTER(CompleteLayout)], C.c_int
        launch = getattr(lib, prefill_symbol)
        launch.argtypes = [C.c_void_p]*10 + [C.c_size_t] + [C.c_void_p]*2 + [C.c_int]*6 + [C.c_void_p]
        launch.restype = C.c_int
        _Q8_PREFILL = (prefill_identity, lib, query, launch)
    prefill_layout = CompleteLayout()
    parent_native.check(_Q8_PREFILL[2](capacity, 256, 2048, 512, 8, C.byref(prefill_layout)), 'Q8 prefill layout')

    # Reuse the original W8-head op and its exact ABI; no dense/routed W4 library is loaded.
    head_path = str(Path(settings['head_library']).resolve(strict=True))
    if 'head' in parent_native._libs and parent_native._libs['head'][0] != head_path:
        raise RuntimeError('Cannot replace the W8 head in a live worker')
    if 'head' not in parent_native._libs:
        lib = C.CDLL(head_path)
        query, launch = lib.ornith_head_i8_tile_get_layout, lib.ornith_head_i8_tile_launch
        query.argtypes, query.restype = [C.c_int]*2+[C.POINTER(parent_native.HeadLayout)], C.c_int
        launch.argtypes = [C.c_void_p]*4+[C.c_size_t]+[C.c_void_p]*2+[C.c_size_t]+[C.c_void_p]+[C.c_int]*4+[C.c_void_p]
        launch.restype = C.c_int
        parent_native._libs['head'] = (head_path, lib, query, launch)
    head = parent_native.HeadLayout()
    parent_native.check(parent_native._libs['head'][2](64, 248320, C.byref(head)), 'W8 head layout')
    if not layout.workspace_bytes or not prefill_layout.workspace_bytes or not head.workspace_bytes:
        raise RuntimeError('Native layout query returned an empty arena')
    return max(layout.workspace_bytes, prefill_layout.workspace_bytes, head.workspace_bytes), layout


@torch.library.custom_op('ornith_ultra::routed_out',
                        mutates_args={'workspace', 'out', 'flags'}, device_types='cuda')
def routed_out(x: torch.Tensor, routes: torch.Tensor, ids: torch.Tensor,
               gate: torch.Tensor, gate_scales: torch.Tensor,
               up: torch.Tensor, up_scales: torch.Tensor,
               down: torch.Tensor, down_scales: torch.Tensor,
               workspace: torch.Tensor, out: torch.Tensor, flags: torch.Tensor,
               capacity: int) -> None:
    parent_native.validate(x, out, capacity, 2048)
    if (_Q8 is None or _Q8_PREFILL is None or routes.shape != (x.shape[0], 8) or ids.shape != routes.shape
            or routes.dtype != torch.float32 or ids.dtype != torch.int32
            or not routes.is_contiguous() or not ids.is_contiguous()
            or routes.device != x.device or ids.device != x.device
            or workspace.device != x.device or flags.device != x.device):
        raise ValueError('Q8 complete operation requires contiguous FP32 top8 weights and I32 IDs on the same device')
    ptr = parent_native.ptr
    binding = _Q8_PREFILL if x.shape[0] >= _Q8_PREFILL[0][3] else _Q8
    parent_native.check(binding[3](ptr(x), ptr(routes), ptr(ids),
        ptr(gate), ptr(gate_scales), ptr(up), ptr(up_scales), ptr(down), ptr(down_scales),
        ptr(workspace), workspace.numel(), ptr(out), ptr(flags), x.shape[0], capacity,
        256, 2048, 512, 8, parent_native.stream(x)), 'Q8 complete routed operation')


@routed_out.register_fake
def _fake(x, routes, ids, gate, gate_scales, up, up_scales, down, down_scales,
          workspace, out, flags, capacity):
    return None
