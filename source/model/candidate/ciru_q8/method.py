import torch
from types import MethodType
from ornith_g256.ultra_method import Q8MoEMethod, _load_q8
from ornith_g256.ultra_contract import FIELDS, target_prefix
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig,FusedMoEQuantDesc
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

def load_model_banks(self,weights):
    # This format stores logical expert keys. The pinned FusedMoE delegates
    # its loader to RoutedExperts while its parameter names remain nested.
    # Adapt names only; the sealed parent loader checks/copies exact payloads.
    def nested():
        for key,value in weights:
            if key not in FIELDS:raise ValueError('Unexpected Ciru Q8 field: '+key)
            yield 'routed_experts.'+key,value
    yield from _load_q8(self,nested())

class Dispatch(FusedMoEQuantConfig):
    def __init__(self):
        super().__init__(_a1=FusedMoEQuantDesc(dtype='ciru_h128_s8_g256',shape=GroupShape(1,256)),
            _a2=FusedMoEQuantDesc(dtype='ciru_h128_s8_g256',shape=GroupShape(1,256)),
            _w1=FusedMoEQuantDesc(dtype='ciru_h128_w8_g256',shape=GroupShape(1,256)),
            _w2=FusedMoEQuantDesc(dtype='ciru_h128_w8_g256',shape=GroupShape(1,256)),is_scale_swizzled=False)
    @property
    def ocp_mx_scheme(self):return None
    def config_name(self,dtype):raise RuntimeError('Ciru Q8 owns explicit native dispatch')

class CiruQ8MoEMethod(Q8MoEMethod):
    def create_weights(self,layer,num_experts,hidden_size,intermediate_size_per_partition,params_dtype,**attrs):
        if (num_experts,hidden_size,intermediate_size_per_partition,params_dtype)!=(256,2048,512,torch.bfloat16):
            raise ValueError('Ciru Q8 requires E256/H2048/I512 BF16')
        if target_prefix(self.prefix) not in {f'layers.{i}.mlp.experts' for i in range(40)}:raise ValueError(self.prefix)
        self.runtime=self.slot=None;layer._ornith_weights_ready=False;layer._ultra_loaded=set()
        for role,n,k in (('gate',512,2048),('up',512,2048),('down',2048,512)):
            for suffix,shape,dtype in (('codes',(256,n//16,k//32,2,16,16),torch.int8),
                                       ('scales',(256,n//16,k//256,16),torch.float16)):
                layer.register_parameter(f'ultra_{role}_{suffix}',torch.nn.Parameter(torch.empty(shape,dtype=dtype),requires_grad=False))
        layer.load_weights=MethodType(load_model_banks,layer)
    def get_fused_moe_quant_config(self,layer):
        if self.moe_quant_config is None:self.moe_quant_config=Dispatch()
        return self.moe_quant_config
