# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ciru. Group query rows sharing one request and KV head.
# Split softmax follows vLLM/SGLang decode attention; native BF16 cache unchanged.
import torch
from vllm.triton_utils import triton,tl
from ciru_q8.attention_partition import _kernels,_metadata

@triton.jit
def shared_query_stage1(Q,K,V,Starts,Seq,Table,Mid,
    QS:tl.constexpr,QH:tl.constexpr,KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,KS3:tl.constexpr,
    VS0:tl.constexpr,VS1:tl.constexpr,VS2:tl.constexpr,TS:tl.constexpr,
    PAGE:tl.constexpr,SPLITS:tl.constexpr,BH:tl.constexpr,BN:tl.constexpr,
    GROUP_Q:tl.constexpr,MAX_Q:tl.constexpr,SCALE:tl.constexpr):
    req=tl.program_id(0);group=tl.program_id(1);split=tl.program_id(2)
    kv=group%2;qt=group//2
    first=tl.load(Starts+req);last=tl.load(Starts+req+1);length=tl.load(Seq+req)
    local=qt*GROUP_Q+tl.arange(0,BH)//8
    head=kv*8+tl.arange(0,BH)%8
    row=first+local;active=(row<last)&(local<MAX_Q)
    causal=length-(last-first)+local+1
    d=tl.arange(0,256);n=tl.arange(0,BN)
    q=tl.load(Q+row[:,None]*QS+head[:,None]*QH+d[None,:],active[:,None],other=0)
    width=tl.cdiv(length,SPLITS);begin=split*width;end=tl.minimum(begin+width,length)
    mx=tl.full((BH,),-float('inf'),tl.float32);den=tl.zeros((BH,),tl.float32)
    acc=tl.zeros((BH,256),tl.float32)
    for base in tl.range(begin,end,BN):
        pos=base+n
        page=tl.load(Table+req*TS+pos//PAGE,pos<end,other=0).to(tl.int64)
        k=tl.load(K+page[None,:]*KS0+kv*KS1+(d[:,None]//8)*KS2+(pos[None,:]%PAGE)*KS3+d[:,None]%8,pos[None,:]<end,other=0)
        scores=tl.dot(q,k)*SCALE
        mask=active[:,None]&(pos[None,:]<end)&(pos[None,:]<causal[:,None])
        scores=tl.where(mask,scores,-float('inf'))
        next_mx=tl.maximum(mx,tl.max(scores,1))
        safe_mx=tl.where(next_mx==-float('inf'),0.,next_mx)
        correction=tl.exp(mx-safe_mx)
        prob=tl.exp(scores-safe_mx[:,None])
        v=tl.load(V+page[:,None]*VS0+kv*VS1+d[None,:]*VS2+pos[:,None]%PAGE,pos[:,None]<end,other=0)
        acc=acc*correction[:,None]+tl.dot(prob.to(v.dtype),v)
        den=den*correction+tl.sum(prob,1);mx=next_mx
    safe_den=tl.where(den>0,den,1.)
    address=((row[:,None]*16+head[:,None])*SPLITS+split)*257+d[None,:]
    tl.store(Mid+address,acc/safe_den[:,None],active[:,None])
    lse=tl.where(den>0,mx+tl.log(safe_den),-float('inf'))
    tl.store(Mid+((row*16+head)*SPLITS+split)*257+256,lse,active)

@triton.jit
def shared_query_reduce(Mid,Out,Starts,Seq,QS:tl.constexpr,QH:tl.constexpr,
    ROWS:tl.constexpr,REQS:tl.constexpr,RB:tl.constexpr,SPLITS:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1)
    rr=tl.arange(0,RB);start=tl.load(Starts+rr,rr<REQS,other=0);end=tl.load(Starts+rr+1,rr<REQS,other=0)
    live=tl.sum(((rr<REQS)&(start<=row)&(row<end)).to(tl.int32),0)>0
    d=tl.arange(0,256);ss=tl.arange(0,SPLITS)
    ls=tl.load(Mid+((row*16+head)*SPLITS+ss)*257+256,live,other=-float('inf'))
    mx=tl.max(ls,0);mx=tl.where(live,mx,0.)
    p=tl.exp(ls-mx);den=tl.sum(p,0);den=tl.where(live,den,1.)
    v=tl.load(Mid+((row*16+head)*SPLITS+ss[:,None])*257+d[None,:],live,other=0.)
    result=tl.sum(p[:,None]*v,0)/den
    tl.store(Out+row*QS+head*QH+d,result)

def forward(query,key_cache,value_cache,output,block_table,query_start_loc,seq_lens,sm_scale,k_scale,v_scale,*,max_query_len=8,block_n=32,group_q=4):
    assert query.shape[1:]==(16,256) and query.dtype==key_cache.dtype==value_cache.dtype==torch.bfloat16
    assert query.stride(-1)==value_cache.stride(-1)==key_cache.stride(-1)==1
    assert query.shape[0]<=128 and 1<=max_query_len<=16 and block_n in (16,32,64) and group_q in (2,4)
    rows=query.shape[0];splits=32
    scratch=torch.empty((rows,16,splits,257),device=query.device,dtype=torch.float32)
    shared_query_stage1[(seq_lens.numel(),2*triton.cdiv(max_query_len,group_q),splits)](
        query,key_cache,value_cache,query_start_loc,seq_lens,block_table,scratch,
        QS=query.stride(0),QH=query.stride(1),KS0=key_cache.stride(0),KS1=key_cache.stride(1),KS2=key_cache.stride(2),KS3=key_cache.stride(3),
        VS0=value_cache.stride(0),VS1=value_cache.stride(1),VS2=value_cache.stride(2),TS=block_table.stride(0),
        PAGE=key_cache.shape[3],SPLITS=splits,BH=group_q*8,BN=block_n,GROUP_Q=group_q,MAX_Q=max_query_len,SCALE=sm_scale,
        num_warps=4,num_stages=1,waves_per_eu=1,matrix_instr_nonkdim=16,kpack=2)
    shared_query_reduce[(rows,16)](scratch,output,query_start_loc,seq_lens,QS=output.stride(0),QH=output.stride(1),ROWS=rows,REQS=seq_lens.numel(),RB=triton.next_power_of_2(seq_lens.numel()),SPLITS=splits,num_warps=4)
    return output
