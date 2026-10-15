"""Original-basis Q8 MoE, BF16 prefill, and selected phase-aware Q8 dense decode."""
from types import MethodType
import torch
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig, FusedMoEQuantDesc
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from .moe_base import OrnithMoEMethodBase
from .ultra_contract import FIELDS, bank_specs, dense_specs, target_prefix


class UltraBF16LinearMethod(LinearMethodBase):
    def __init__(self, prefix):
        self.prefix = prefix
        self.n, self.k, self.precision_origin = dense_specs()[target_prefix(prefix)]

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **attrs):
        if (input_size_per_partition != self.k or input_size != self.k
                or sum(output_partition_sizes) != self.n or output_size != self.n
                or params_dtype != torch.bfloat16 or getattr(layer, 'tp_size', 1) != 1
                or getattr(layer, 'has_bias', False)):
            raise ValueError(f'Unsupported ULTRA dense geometry: {self.prefix}')
        layer._ultra_dense_loaded = layer._ultra_dense_ready = False
        parameter = torch.nn.Parameter(torch.empty((self.n, self.k), dtype=torch.bfloat16), requires_grad=False)
        layer.register_parameter('ultra_weight', parameter)

        def load(destination, value, *shard_args):
            if destination is not parameter or layer._ultra_dense_loaded or layer._ultra_dense_ready:
                raise ValueError('Duplicate/late ULTRA dense weight load')
            if value.shape != parameter.shape or value.dtype != torch.bfloat16:
                raise ValueError('ULTRA dense full-bank shape/dtype mismatch')
            if destination.device.type == 'meta' or value.device.type == 'meta':
                raise ValueError('ULTRA loader requires materialized storage')
            # Only ultra_weight records are accepted; each explicitly stores the full fused bank.
            with torch.no_grad():
                destination.copy_(value)
            layer._ultra_dense_loaded = True
        parameter.weight_loader = load

    def process_weights_after_loading(self, layer):
        if not layer._ultra_dense_loaded:
            raise ValueError(f'Missing ULTRA dense matrix: {self.prefix}')
        layer._ultra_dense_ready = True

    def apply(self, layer, x, bias=None):
        if (not layer._ultra_dense_ready or bias is not None or x.dtype != torch.bfloat16
                or x.shape[-1] != self.k or x.device != layer.ultra_weight.device):
            raise RuntimeError('ULTRA dense requires a loaded BF16 bank and BF16 input')
        shape = x.shape[:-1]
        x = x.reshape(-1, self.k).contiguous()
        out = torch.empty((x.shape[0], self.n), dtype=x.dtype, device=x.device)
        torch.mm(x, layer.ultra_weight.t(), out=out)
        return out.view(*shape, self.n)



class UltraQ8DecodeLinearMethod(UltraBF16LinearMethod):
    """Selected GDN input: one Q8 G32 bank, BF16 prompt and native decode math."""
    def __init__(self, prefix):
        from .ultra_dense_contract import SELECTED
        if target_prefix(prefix) not in SELECTED:
            raise ValueError('Q8 dense decode is restricted to30 GDN input projections')
        super().__init__(prefix)
        self.runtime = self.slot = None

    def apply(self, layer, x, bias=None):
        if (self.runtime is None or not layer._ultra_dense_ready
                or not layer._ultra_dense_shadow_ready or bias is not None
                or layer._ultra_dense_prefill_policy != 'gdn_input_coherent_q8_g32_v2'
                or x.dtype != torch.bfloat16 or x.shape[-1] != self.k):
            raise RuntimeError('Selected dense projection requires its bound BF16 and Q8 banks')
        from .ultra_dense_native import dense_phase_out
        shape = x.shape[:-1]
        x = x.reshape(-1, self.k).contiguous()
        out = torch.empty((x.shape[0], self.n), dtype=x.dtype, device=x.device)
        dense_phase_out(x, layer.ultra_weight, layer.ultra_dense_codes,
                        layer.ultra_dense_scales, out, self.runtime.slots[self.slot], self.runtime.dense_live_rows)
        return out.view(*shape, self.n)


def _load_q8(self, weights):
    if self._ornith_weights_ready:
        raise ValueError('Q8 banks are already sealed')
    for key, value in weights:
        prefix = 'routed_experts.'
        if not key.startswith(prefix) or key[len(prefix):] not in FIELDS:
            raise ValueError(f'Unexpected Q8 bank field: {key}')
        field = key[len(prefix):]
        if field in self._ultra_loaded:
            raise ValueError('Duplicate Q8 field: ' + field)
        param = self._parameters[field]
        if value.shape != param.shape or value.dtype != param.dtype:
            raise ValueError('Q8 bank shape/dtype mismatch: ' + field)
        if value.device.type != 'cpu' or param.device.type == 'meta':
            raise ValueError('Q8 checkpoint admission expects materialized CPU source tensors')
        if field.endswith('_scales') and not bool(torch.isfinite(value).all()):
            raise ValueError('Q8 scale bits must encode finite FP16 values')
        # Same dtype copy preserves the FP16 scale payload. No affine offsets/H128.
        with torch.no_grad():
            param.copy_(value)
        self._ultra_loaded.add(field)
        yield prefix + field


class UltraDispatchConfig(FusedMoEQuantConfig):
    def __init__(self):
        super().__init__(
            _a1=FusedMoEQuantDesc(dtype='ornith_ultra_native', shape=GroupShape(1, 32)),
            _a2=FusedMoEQuantDesc(dtype='ornith_ultra_native', shape=GroupShape(1, 32)),
            _w1=FusedMoEQuantDesc(dtype='ornith_q8_0', shape=GroupShape(1, 32)),
            _w2=FusedMoEQuantDesc(dtype='ornith_q8_0', shape=GroupShape(1, 32)),
            is_scale_swizzled=False)

    @property
    def ocp_mx_scheme(self): return None

    def config_name(self, dtype):
        raise RuntimeError('ULTRA uses explicit native complete-operation dispatch')


class Q8MoEMethod(OrnithMoEMethodBase):
    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        if (num_experts, hidden_size, intermediate_size_per_partition, params_dtype) != (256, 2048, 512, torch.bfloat16):
            raise ValueError('ULTRA routed banks require E256/H2048/I512 and BF16 model inputs')
        if target_prefix(self.prefix) not in {f'layers.{i}.mlp.experts' for i in range(40)}:
            raise ValueError('ULTRA accepts only target routed layers0..39')
        self.runtime = self.slot = None
        layer._ornith_weights_ready = False
        layer._ultra_loaded = set()
        for field, (shape, dtype) in bank_specs().items():
            layer.register_parameter(field, torch.nn.Parameter(torch.empty(shape,
                dtype=torch.int8 if dtype == 'I8' else torch.float16), requires_grad=False))
        layer.load_weights = MethodType(_load_q8, layer)

    def process_weights_after_loading(self, layer):
        if layer._ultra_loaded != set(FIELDS):
            raise ValueError(f'Incomplete Q8 bank coverage: {self.prefix}')
        layer._ornith_weights_ready = True

    def get_fused_moe_quant_config(self, layer):
        if self.moe_quant_config is None:
            self.moe_quant_config = UltraDispatchConfig()
        return self.moe_quant_config

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        if self.runtime is None or not layer._ornith_weights_ready:
            raise RuntimeError('Q8 routed weights require loaded banks and a bound worker')
        from .ultra_native import routed_out
        out = torch.empty_like(x)
        routed_out(x, topk_weights, topk_ids,
            *(getattr(layer, name) for name in FIELDS), self.runtime.workspace,
            out, self.runtime.slots[self.slot], self.runtime.capacity)
        return out
