"""Model-independent Q8 expert adapter for the retained Ciru native kernels."""
import importlib.abc, importlib.util, json, pathlib, sys

_REGISTERED=False

def register():
    global _REGISTERED
    if _REGISTERED:return
    # Preserve the parent's pinned correctness fixes before vLLM imports them.
    import ornith_g256
    from vllm.model_executor.layers.quantization import register_quantization_config
    from .config import CiruQ8Config
    register_quantization_config('ciru_q8')(CiruQ8Config)
    root=pathlib.Path(__file__).resolve().parents[1]/'upstream/vllm'
    class ModelSources(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in ('vllm.model_executor.models.minicpmv',
                'vllm.model_executor.models.minicpmv4_6','vllm.model_executor.models.minicpmv4_7'):return None
            source=root/(fullname.replace('.','/')+'.py')
            return importlib.util.spec_from_file_location(fullname,source)
    sys.meta_path.insert(0,ModelSources())
    from vllm.model_executor.models import ModelRegistry
    ModelRegistry.register_model('MiniCPMV4_7ForConditionalGeneration',
        'vllm.model_executor.models.minicpmv4_7:MiniCPMV4_7ForConditionalGeneration')
    from .mtp_prefix import install
    install()
    _REGISTERED=True
