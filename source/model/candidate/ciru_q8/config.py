import torch
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
CONTRACT=dict(quant_method='ciru_q8',checkpoint_format='ciru_h128_w8_g256_bf16_v1',
              group_size=256,transform_block=128,routed_weight_bits=8)
class CiruQ8Config(QuantizationConfig):
    def __init__(self,config):
        super().__init__()
        if any(config.get(k)!=v for k,v in CONTRACT.items()):raise ValueError('Ciru Q8 checkpoint contract mismatch')
        self.checkpoint_config=dict(config)
    @classmethod
    def get_name(cls):return 'ciru_q8'
    def get_supported_act_dtypes(self):return [torch.bfloat16]
    @classmethod
    def get_min_capability(cls):return 0
    @staticmethod
    def get_config_filenames():return []
    @classmethod
    def from_config(cls,config):return cls(config)
    def get_quant_method(self,layer,prefix):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        if isinstance(layer,RoutedExperts):
            if prefix.startswith('mtp.'):
                from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
                return UnquantizedFusedMoEMethod(layer.moe_config)
            from .method import CiruQ8MoEMethod
            CiruQ8MoEMethod.validate_layer_scope(layer)
            return CiruQ8MoEMethod(layer.moe_config,self,prefix)
        # Retain original BF16 operators and their stacked checkpoint loaders.
        from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, UnquantizedEmbeddingMethod
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, VocabParallelEmbedding):
            return UnquantizedEmbeddingMethod()
        return None
