"""One selected Q8 weight bank for both phases, with BF16 prefill operands."""
import ctypes as C
import hashlib
from pathlib import Path
import torch
from . import native as parent_native
from .ultra_dense_contract import DenseLayout, SMALL_SYMBOL, LARGE_SYMBOL, validate_settings
from .ultra_dense_phase import segments_for, capture_live_pointer

_BINDINGS = None


def configure(settings):
    global _BINDINGS
    validate_settings(settings)
    identities = []
    for label, symbol in (('small', SMALL_SYMBOL), ('large', LARGE_SYMBOL)):
        path = Path(settings[f'q8_dense_{label}_library']).resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != settings[f'q8_dense_{label}_library_sha256']:
            raise ValueError('Dense native library hash mismatch')
        identities.append((str(path), digest, symbol))
    if _BINDINGS is not None:
        if tuple(b[0] for b in _BINDINGS) != tuple(identities):
            raise RuntimeError('Cannot replace dense native bindings in a live worker')
        return
    bindings = []
    for identity in identities:
        lib = C.CDLL(identity[0])
        query = lib.ornith_q8_dense_bf16_get_layout
        query.argtypes, query.restype = [C.c_int]*3 + [C.POINTER(DenseLayout)], C.c_int
        launch = getattr(lib, identity[2])
        launch.argtypes, launch.restype = [C.c_void_p]*6 + [C.c_int]*3 + [C.c_void_p], C.c_int
        for rows in (1, 64):
            layout = DenseLayout()
            parent_native.check(query(rows, 12288, 2048, C.byref(layout)), 'Dense Q8 layout')
            if tuple(getattr(layout, k) for k, _ in layout._fields_) != (rows*2048*2, 25165824, 1572864, rows*12288*2, 0):
                raise ValueError('Dense native layout mismatch')
        bindings.append((identity, lib, query, launch))
    _BINDINGS = tuple(bindings)


@torch.library.custom_op('ornith_ultra::dense_phase_out', mutates_args={'out', 'flags'}, device_types='cuda')
def dense_phase_out(x: torch.Tensor, bf16: torch.Tensor, codes: torch.Tensor,
                    scales: torch.Tensor, out: torch.Tensor, flags: torch.Tensor, live_rows: torch.Tensor) -> None:
    if (_BINDINGS is None or x.dtype != torch.bfloat16 or bf16.dtype != x.dtype or out.dtype != x.dtype
            or tuple(bf16.shape) != (12288, 2048) or x.ndim != 2 or x.shape[1] != 2048
            or tuple(out.shape) != (x.shape[0], 12288)
            or tuple(codes.shape) != (768, 64, 2, 16, 16) or codes.dtype != torch.int8
            or tuple(scales.shape) != (768, 64, 16) or scales.dtype != torch.float16
            or flags.dtype != torch.int32 or flags.numel() != 1
            or live_rows.dtype != torch.int32 or tuple(live_rows.shape) != (1,)
            or any(not t.is_contiguous() or t.device != x.device for t in (x, bf16, codes, scales, out, flags, live_rows))):
        raise ValueError('Invalid selected GDN-input dense operands')
    ptr = parent_native.ptr
    live_pointer = capture_live_pointer(live_rows)
    for first, end, prefill in segments_for(x.shape[0]):
        if prefill:
            torch.mm(x[first:end], bf16.t(), out=out[first:end])
        else:
            for begin in range(first, end, 64):
                stop = min(begin + 64, end)
                rows = stop - begin
                binding = _BINDINGS[0 if rows <= 16 else 1]
                parent_native.check(binding[3](ptr(x[begin:stop]), ptr(codes), ptr(scales),
                    ptr(out[begin:stop]), ptr(flags), live_pointer, rows, 12288, 2048, parent_native.stream(x)),
                    'Dense Q8 decode projection')


@dense_phase_out.register_fake
def _fake(x, bf16, codes, scales, out, flags, live_rows):
    return None


def load_shadow(layer, root, bank):
    for field, dtype, shape in (('codes', torch.int8, bank['code_shape']),
                                ('scales', torch.float16, bank['scale_shape'])):
        path = root / bank['bank_directory'] / bank[field + '_file']
        data = bytearray(path.read_bytes())
        if len(data) != bank[field + '_bytes'] or hashlib.sha256(data).hexdigest() != bank[field + '_sha256']:
            raise ValueError('Dense shadow bank hash/extent mismatch: ' + str(path))
        value = torch.frombuffer(data, dtype=dtype).view(shape).to(layer.ultra_weight.device)
        layer.register_buffer('ultra_dense_' + field, value, persistent=False)
    # Retain the fast native decode kernels. Reuse the existing BF16 parameter
    # storage for the exact same dequantized shadow during prefill. Small row
    # tiles bound scratch memory; the original checkpoint is never modified.
    with torch.no_grad():
        for first in range(0, 768, 32):
            stop = min(first + 32, 768)
            rows = (stop - first) * 16
            codes = layer.ultra_dense_codes[first:stop].permute(0, 3, 1, 2, 4).reshape(rows, 2048)
            scales = layer.ultra_dense_scales[first:stop].permute(0, 2, 1).reshape(rows, 64)
            weight = (codes.float() * scales.float().repeat_interleave(32, dim=1)).bfloat16()
            layer.ultra_weight[first*16:stop*16].copy_(weight)
    layer._ultra_dense_prefill_policy = 'gdn_input_coherent_q8_g32_v2'
    layer._ultra_dense_shadow_ready = True
