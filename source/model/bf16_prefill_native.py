"""Native PR172 BF16/LDS adapter; retained gather and output ownership."""
import ctypes as C
import hashlib
import pathlib
import torch

class NativePrefill:
    def __init__(self):
        root=pathlib.Path(__file__).resolve().parent/'bf16-prefill-native-v1'
        self.path=root/'prefill.so';self.lib=C.CDLL(str(self.path))
        self.fn=self.lib.bf16_prefill_native
        self.fn.argtypes=[C.c_void_p]*7+[C.c_int]*2+[C.c_int64]*10+[C.c_float,C.c_void_p]
        self.fn.restype=C.c_int
        self.identity=dict(options=dict(native=True,query_columns=64,grouped_heads=8,query_tokens=8,BLOCK_N=32,num_warps=8),compiled=True,
            library=str(self.path),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256((root/'prefill.hip').read_bytes()).hexdigest())
    def __call__(self,**kw):
        assert kw['layout']=='thd' and kw['causal'] and kw['dropout_p']==0
        assert all(kw[k] is None for k in ['bias','alibi_slopes','sd_mask','q_descale','k_descale','v_descale'])
        q,k,v,o,lse=(kw[x] for x in ('q','k','v','o','softmax_lse'))
        assert q.shape[1:]==(16,256) and k.shape[1:]==v.shape[1:]==(2,256)
        assert all(x.dtype==torch.bfloat16 and x.stride(-1)==1 for x in (q,k,v,o))
        qs,ks=kw['cu_seqlens_q'],kw['cu_seqlens_k']
        code=self.fn(*[x.data_ptr() for x in (q,k,v,o,lse,qs,ks)],kw['max_seqlens_q'],qs.numel()-1,
                     q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1),
                     o.stride(0),o.stride(1),lse.stride(0),lse.stride(1),kw['sm_scale'],torch.cuda.current_stream().cuda_stream)
        assert code==0,('HIP native prefill launch',code)
        return o,lse
    def receipt(self):return self.identity
