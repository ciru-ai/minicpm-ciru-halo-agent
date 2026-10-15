"""Isolated API entrypoint: installed runtime and existing profiles stay intact."""
import os,runpy
for key,value in {'VLLM_DISABLE_SHARED_EXPERTS_STREAM':'1','VLLM_ROCM_USE_AITER_MOE':'0',
                  'VLLM_ROCM_USE_SKINNY_GEMM':'0','VLLM_USE_V2_MODEL_RUNNER':os.environ.get('CIRU_DSPARK_RUNNER','1'),
                  'VLLM_ENABLE_V1_MULTIPROCESSING':'0','VLLM_ROCM_USE_AITER':'1'}.items():os.environ[key]=value
import ciru_q8
ciru_q8.register()
if __name__ == '__main__':
    if os.environ.get('CIRU_TOKENIZE_FIX') == '1':
        from tokenize_alias import install
        install()
    runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')
