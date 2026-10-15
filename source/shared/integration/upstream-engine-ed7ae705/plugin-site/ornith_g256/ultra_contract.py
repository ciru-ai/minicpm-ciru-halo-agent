"""Torch-free checkpoint/ABI contract for the isolated mixed-Q8 candidate."""
import ctypes as C
import re

FORMAT = 'ornith_ultra_q8_v1'
QUANT_CONFIG = dict(quant_method='ornith_g256', checkpoint_format=FORMAT,
                    routed_weight_bits=8, group_size=32, transform_block=0,
                    dense_dtype='bfloat16', head_format='w8_g128_h128')
FIELDS = tuple(f'ultra_{projection}_{part}' for projection in ('gate', 'up', 'down')
               for part in ('codes', 'scales'))
NUMERIC_WARMUP_MASK = 0x8 | 0x10 | 0x200 | 0x400


class CompleteLayout(C.Structure):
    _fields_ = [(name, C.c_size_t) for name in (
        'workspace_bytes', 'counts', 'offsets', 'cursors', 'routes', 'descriptors',
        'queue', 'gate', 'up', 'middle', 'route_out', 'final_f32', 'descriptor_capacity')]


def bank_specs():
    out = {}
    for projection, n, k in (('gate', 512, 2048), ('up', 512, 2048), ('down', 2048, 512)):
        out[f'ultra_{projection}_codes'] = ((256, n//16, k//32, 2, 16, 16), 'I8')
        out[f'ultra_{projection}_scales'] = ((256, n//16, k//32, 16), 'F16')
    return out


def dense_specs():
    """Native method prefixes -> (N,K,precision provenance), exactly160."""
    out = {}
    for layer in range(40):
        root = f'layers.{layer}.'
        out[root+'mlp.shared_expert.gate_up_proj'] = (1024, 2048, 'original_bf16')
        out[root+'mlp.shared_expert.down_proj'] = (2048, 512, 'original_bf16')
        if layer % 4 != 3:
            out[root+'linear_attn.in_proj_qkvz'] = (12288, 2048, 'original_bf16')
            out[root+'linear_attn.out_proj'] = (2048, 4096, 'q8_origin_bf16')
        else:
            out[root+'self_attn.qkv_proj'] = (9216, 2048, 'q8_origin_bf16')
            out[root+'self_attn.o_proj'] = (2048, 4096, 'q8_origin_bf16')
    return out


def target_prefix(prefix):
    if prefix.startswith('mtp.') or '.mtp.' in prefix:
        raise ValueError('Native MTP is not admitted to the target ULTRA bank')
    match = re.search(r'(?:^|\.)(layers\.[0-9]+\..+)$', prefix)
    if not match:
        raise ValueError(f'Not a target layer prefix: {prefix}')
    return match.group(1)


def checkpoint_dense_key(prefix):
    prefix = target_prefix(prefix)
    if prefix not in dense_specs():
        raise ValueError(f'Unexpected ULTRA dense prefix: {prefix}')
    # The installed Qwen3.5 mapper turns this proven full-bank alias into qkvz.
    prefix = prefix.replace('.linear_attn.in_proj_qkvz', '.linear_attn.in_proj_qkv')
    return 'model.language_model.' + prefix + '.ultra_weight'


def validate_settings(settings):
    if settings.get('checkpoint_format') != FORMAT:
        raise ValueError('ULTRA checkpoint requires the explicit matching runtime format')
    for name in ('q8_library', 'q8_library_sha256', 'q8_kernel', 'head_library'):
        if not settings.get(name):
            raise ValueError(f'ULTRA runtime requires {name}')
    if settings['q8_kernel'] not in ('i8', 'f16', 'cooperative', 'cooperative_hybrid'):
        raise ValueError('Q8 kernel must be i8, f16, cooperative, or cooperative_hybrid, explicitly selected')
    sha = settings['q8_library_sha256']
    if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Q8 library SHA256 must identify the selected binary')
    for name in ('q8_prefill_library', 'q8_prefill_library_sha256',
                 'q8_prefill_symbol', 'q8_prefill_min_rows'):
        if not settings.get(name):
            raise ValueError(f'ULTRA v10 runtime requires {name}')
    if (settings['q8_kernel'] != 'cooperative_hybrid'
            or settings['q8_prefill_symbol'] != 'ornith_q8_complete_cooperative_m64n64_launch'
            or settings['q8_prefill_min_rows'] != 2048):
        raise ValueError('ULTRA v10 requires primary v7 hybrid and M64 prefill starting at2048 rows')
    sha = settings['q8_prefill_library_sha256']
    if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Q8 prefill library SHA256 must identify the selected binary')
    if any(settings.get(name) for name in ('routed_decode_n32', 'routed_storage_n32')):
        raise ValueError('W4 N32 layouts are not valid for ULTRA')
    if any(name in settings for name in ('routed_activation_bits', 'routed_prefill_activation_bits')):
        raise ValueError('Remove W4 routed precision settings from ULTRA settings')
    if settings.get('attention_mode') != 'stock_rocm':
        raise ValueError('Initial ULTRA integration preserves the stock_rocm attention profile')
