"""Strict target coverage and a shared Q8/head arena, bound before capture."""
from types import SimpleNamespace
import torch
import vllm.envs as envs
from vllm.logger import init_logger
from .method import G256LinearMethod, G256MoEMethod, W8HeadMethod
from .runtime import Runtime as ParentRuntime
from .ultra_method import UltraBF16LinearMethod, UltraQ8DecodeLinearMethod, Q8MoEMethod
from .ultra_contract import bank_specs, dense_specs, target_prefix
from .ultra_native import configure


def inspect_layers(model):
    routed, dense, heads = [], [], []
    for name, layer in model.named_modules():
        method = getattr(layer, 'quant_method', None)
        if isinstance(method, Q8MoEMethod): routed.append((name, layer))
        elif isinstance(method, UltraBF16LinearMethod): dense.append((name, layer))
        elif isinstance(method, W8HeadMethod): heads.append((name, layer))
        elif isinstance(method, (G256LinearMethod, G256MoEMethod)):
            raise ValueError(f'W4 target method remains in ULTRA: {name}')
    expected_dense = dense_specs()
    actual_dense = {target_prefix(layer.quant_method.prefix): (layer.quant_method.n, layer.quant_method.k,
        layer.quant_method.precision_origin) for _, layer in dense}
    expected_routed = {f'layers.{l}.mlp.experts' for l in range(40)}
    actual_routed = {target_prefix(layer.quant_method.prefix) for _, layer in routed}
    if ((len(routed), len(dense), len(heads)) != (40, 160, 1)
            or actual_dense != expected_dense or actual_routed != expected_routed):
        raise ValueError(f'ULTRA requires exact40 Q8/160 BF16/1 W8 coverage; got {len(routed)}/{len(dense)}/{len(heads)}')
    for name, layer in dense:
        if not layer._ultra_dense_ready or layer.ultra_weight.dtype != torch.bfloat16:
            raise ValueError(f'Unloaded BF16 dense matrix: {name}')
    for name, layer in routed:
        if not layer._ornith_weights_ready:
            raise ValueError(f'Unloaded Q8 routed layer: {name}')
        for field, (shape, dtype) in bank_specs().items():
            value = getattr(layer, field)
            if tuple(value.shape) != shape or value.dtype != (torch.int8 if dtype == 'I8' else torch.float16):
                raise ValueError(f'Invalid Q8 bank: {name}.{field}')
    return routed, dense, heads


class Runtime(ParentRuntime):
    def __init__(self, model, settings, capacity, device, *, max_model_len=4096, max_num_seqs=8):
        if not envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            raise ValueError('ULTRA shared arena requires VLLM_DISABLE_SHARED_EXPERTS_STREAM=1')
        self.backend = SimpleNamespace(device=torch.device(device))
        self.capacity = capacity
        self.head_geometry = settings.get('head_geometry', 2)
        if self.head_geometry not in (0, 1, 2):
            raise ValueError('Native head geometry must be0,1,2')
        routed, dense, heads = inspect_layers(model)
        from .ultra_dense_contract import SELECTED, SELECTED_BYTES, read_manifest
        from .ultra_dense_native import configure as configure_dense, load_shadow
        selected = [(name, layer) for name, layer in dense
                    if isinstance(layer.quant_method, UltraQ8DecodeLinearMethod)]
        if ({target_prefix(layer.quant_method.prefix) for _, layer in selected} != SELECTED
                or len(selected) != 30 or sum(type(layer.quant_method) is UltraBF16LinearMethod for _, layer in dense) != 130):
            raise ValueError('V13 requires exactly30 selected and130 untouched BF16 dense methods')
        root, banks = read_manifest(settings)
        configure_dense(settings)
        for _, layer in selected:
            load_shadow(layer, root, banks[target_prefix(layer.quant_method.prefix)])
        layers = routed + heads + selected
        for name, layer in layers:
            ready = (layer._ornith_weights_ready if isinstance(layer.quant_method, Q8MoEMethod)
                     else layer._ultra_dense_shadow_ready if isinstance(layer.quant_method, UltraQ8DecodeLinearMethod)
                     else layer._g256_ready)
            if not ready:
                raise ValueError(f'Unready native layer: {name}')
            if layer.quant_method.runtime is not None:
                raise ValueError('ULTRA native layers are bound only once')
        size, self.q8_layout = configure(settings, capacity)
        self.workspace = torch.empty(size, dtype=torch.uint8, device=device)
        if self.workspace.data_ptr() % 256:
            raise ValueError('Q8 arena requires256-byte alignment')
        self.error_flags = torch.zeros(len(layers), dtype=torch.int32, device=device)
        self.dense_live_rows = torch.full((1,), -1, dtype=torch.int32, device=device)
        from .ultra_dense_phase import bind_live_rows
        bind_live_rows(self.dense_live_rows)
        self.slots = tuple(self.error_flags[i:i+1] for i in range(len(layers)))
        self.layer_prefixes = tuple(name for name, _ in layers)
        self.layers = tuple(layer for _, layer in layers)
        self.shapes = sorted({(layer.quant_method.n, layer.quant_method.k) for _, layer in dense})
        self.dense_layers = tuple(layer for _, layer in dense)
        for slot, layer in enumerate(self.layers):
            layer.quant_method.runtime, layer.quant_method.slot = self, slot
        init_logger('vllm.ornith_ultra').info(
            'ULTRA v21 bound40 Q8 routed,130 BF16 dense,30 coherent-Q8 GDN input,1 original W8 head; '
            'flags=71; shadow_bytes=%d; kernel=%s; arena=%d', SELECTED_BYTES, settings['q8_kernel'], size)
