"""Immutable v13 dense shadow admission; metadata checks need no Torch/GPU."""
import ctypes as C
import hashlib
import json
from pathlib import Path

SHADOW_SHA256 = '60384b3bd10e163bd1f95f7b987485f7cdd1c2a4f3b29baefa5fbb16fc2ca0f4'
SMALL_SHA256 = '2d1408cbdcbddf63f872473c6eba363e5c63f473c64ed18bad014f66888dd4ca'
LARGE_SHA256 = '33703f7f1157757d5ea904c55a13d6f2fb0b260f8800216a7a272fa35694e72d'
SMALL_SYMBOL = 'ornith_q8_dense_bf16_live_launch'
LARGE_SYMBOL = 'ornith_q8_dense_bf16_m64n64_live_launch'
POLICY = 'gdn_input_coherent_q8_g32_v2'
NUMERIC_WARMUP_MASK = 0x38
SELECTED = frozenset(f'layers.{l}.linear_attn.in_proj_qkvz' for l in range(40) if l % 4 != 3)
SELECTED_BYTES = 802160640


class DenseLayout(C.Structure):
    _fields_ = [(name, C.c_size_t) for name in
                ('input_bytes', 'codes_bytes', 'scale_bytes', 'output_bytes', 'workspace_bytes')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_settings(settings):
    expected = dict(q8_dense_shadow_manifest_sha256=SHADOW_SHA256,
                    q8_dense_small_library_sha256=SMALL_SHA256,
                    q8_dense_large_library_sha256=LARGE_SHA256,
                    q8_dense_small_max_rows=16, q8_dense_max_rows=64, q8_dense_policy=POLICY)
    if any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError('V21 requires coherent selected dense policy and pinned binary/manifest identities')
    for key in ('q8_dense_shadow_root', 'q8_dense_small_library', 'q8_dense_large_library'):
        if not settings.get(key) or not Path(settings[key]).is_absolute():
            raise ValueError('V13 requires an explicit absolute path: ' + key)


def read_manifest(settings, checkpoint=None):
    validate_settings(settings)
    root = Path(settings['q8_dense_shadow_root']).resolve(strict=True)
    manifest_path = root / 'manifest.json'
    if digest(manifest_path) != SHADOW_SHA256:
        raise ValueError('Dense shadow root manifest hash differs')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('status') != 'PASS' or manifest.get('matrix_count') != 160
            or manifest.get('layers') != list(range(40)) or not manifest.get('native_mtp_excluded')
            or manifest.get('format') != 'ornith_dense_q8_shadow_g32_v1'
            or manifest.get('total_shadow_bytes') != 1492910080):
        raise ValueError('Dense shadow inventory differs')
    receipt_path = Path(manifest['checkpoint_receipt']).resolve(strict=True)
    if checkpoint is not None and receipt_path != (Path(checkpoint) / 'ultra-manifest.json').resolve(strict=True):
        raise ValueError('Dense shadow source is a different checkpoint')
    if digest(receipt_path) != manifest['checkpoint_receipt_sha256']:
        raise ValueError('Dense shadow checkpoint receipt hash differs')
    receipt = json.loads(receipt_path.read_text())
    original = {r['key']: r for r in receipt['records']}
    selected = {}
    for bank in manifest['banks']:
        prefix = bank['native_prefix']
        if prefix not in SELECTED:
            continue
        if prefix in selected:
            raise ValueError('Duplicate selected dense shadow')
        if (bank['status'] != 'PASS' or bank['shape'] != [12288, 2048]
                or bank['code_shape'] != [768, 64, 2, 16, 16]
                or bank['scale_shape'] != [768, 64, 16]
                or bank['codes_bytes'] != 25165824 or bank['scales_bytes'] != 1572864
                or bank['codes_file'] != 'codes.i8.bin' or bank['scales_file'] != 'scales.f16.bin'
                or bank['precision_origin'] != 'new_symmetric_g32_q8_from_original_bf16'
                or not bank['packed_roundtrip_all_bytes_equal']
                or bank['baseline_bf16_sha256'] != bank['baseline_bf16']['sha256']
                or bank['baseline_bf16_sha256'] != original[bank['checkpoint_key']]['copied_payload_sha256']):
            raise ValueError('Dense shadow shape/source contract differs: ' + prefix)
        directory = (root / bank['bank_directory']).resolve(strict=True)
        if not directory.is_relative_to(root):
            raise ValueError('Dense shadow escapes immutable root')
        if json.loads((directory / 'manifest.json').read_text()) != bank:
            raise ValueError('Dense shadow leaf differs from pinned root')
        selected[prefix] = bank
    if selected.keys() != SELECTED or sum(b['codes_bytes'] + b['scales_bytes'] for b in selected.values()) != SELECTED_BYTES:
        raise ValueError('V13 requires exactly30 GDN input shadows')
    return root, selected
