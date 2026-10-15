"""BF16 prefill child of the preserved three-stage GDN/shared16 winner."""
import hashlib, inspect, json, math, os, pathlib
import torch
from gdn_spatial_worker import Worker as Parent
from gdn_common import clone_layout, describe
from bf16_prefill_kernel import PARENT_SHA
from bf16_prefill_native import NativePrefill

class Worker(Parent):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        from ciru_q8 import compact_prefill as compact
        mode = os.environ['CIRU_BF16_PREFILL_MODE']
        assert mode in ('capture', 'audit', 'clean', 'parent')
        self._bf_mode = mode; self._bf_ready = False; self._bf_ordinal = 0
        self._bf_done = set(); self._bf_records = []; self._bf_shapes = set()
        self._bf_calls = 0; self._bf_profiled = set(); self._bf_roctx = None
        self._bf_stats = torch.zeros(6, device=self.device, dtype=torch.float64) if mode == 'audit' else None
        parent = compact._prefill
        child = self._bf_fixed = NativePrefill()
        ordinary = compact.try_forward
        def prefill(**kw):
            if not self._bf_ready or mode in ('capture', 'parent'):
                return parent(**kw)
            reqs = kw['cu_seqlens_q'].numel()-1
            bucket = 'c1' if reqs == 1 else 'concurrent'
            trace = self._bf_roctx is not None and bucket not in self._bf_profiled and kw['q'].shape[0]>=4096
            if trace:
                torch.cuda.synchronize(); self._bf_roctx.resume()
                depth = self._bf_roctx.push('BF16_NATIVE_'+bucket)
            try:
                value = child(**kw)
            finally:
                if trace:
                    torch.cuda.synchronize(); self._bf_roctx.pop(depth); self._bf_roctx.pause()
            self._bf_calls += 1
            signature = (kw['q'].shape[0], reqs, tuple(kw['q'].stride()), kw['k'].untyped_storage().nbytes())
            if signature not in self._bf_shapes:
                self._bf_shapes.add(signature)
                with (self.run/'bf16-prefill-shapes.jsonl').open('a') as f:
                    f.write(json.dumps(dict(signature=signature,inputs=describe(kw),compiled=self._bf_fixed.receipt()),default=str)+'\n')
            if trace:
                self._bf_profiled.add(bucket)
                with (self.run/'bf16-prefill-native-selected.jsonl').open('a') as f:
                    f.write(json.dumps(dict(marker='BF16_NATIVE_'+bucket,inputs=describe(kw),compiled=self._bf_fixed.receipt()),default=str)+'\n')
            if self._bf_stats is not None:
                ref = torch.empty_strided(kw['o'].shape, kw['o'].stride(), dtype=kw['o'].dtype, device=kw['o'].device)
                parent(**{**kw, 'o':ref, 'softmax_lse':torch.empty_like(kw['softmax_lse'])})
                live = torch.arange(kw['q'].shape[0],device=self.device)<kw['cu_seqlens_q'][-1]
                active = live[:,None,None]
                diff = torch.where(active, kw['o'].float()-ref.float(), 0.)
                st = self._bf_stats
                st[0].add_((active & (kw['o']!=ref)).sum())
                st[1].add_((active & (~kw['o'].isfinite() | ~ref.isfinite())).sum())
                st[2].copy_(torch.maximum(st[2], diff.abs().amax().double()))
                st[3].add_(diff.double().square().sum())
                st[4].add_(live.sum()*4096); st[5].add_(1)
            return value
        compact._prefill = prefill
        def capture(q,k,v,o,table,starts,lens,**options):
            used = ordinary(q,k,v,o,table,starts,lens,**options)
            if not used or not self._bf_ready or mode != 'capture': return used
            ordinal = self._bf_ordinal; self._bf_ordinal += 1
            if ordinal not in (0,9): return used
            rows=q.shape[0];reqs=lens.numel()
            cohort='c1' if reqs==1 else 'concurrent'
            bucket='long' if rows>=4096 else 'tail' if 16<rows<=2048 else None
            if bucket is None or options['max_seq_len'] < (50000 if reqs==1 else 4096): return used
            key=(cohort,bucket,ordinal)
            if key in self._bf_done:return used
            assert not torch.cuda.is_current_stream_capturing()
            seq=lens.cpu();beg=starts.cpu();tab=table.cpu();page=k.shape[3]
            pages=sorted({int(x) for i,n in enumerate(seq.tolist()) if beg[i+1]>beg[i] for x in tab[i,:math.ceil(n/page)].tolist()})
            index=torch.tensor(pages,device=k.device,dtype=torch.long)
            def meta(t):
                return {**describe(t),'storage_elements':t.untyped_storage().nbytes()//t.element_size()}
            metadata=dict(cohort=cohort,bucket=bucket,ordinal=ordinal,scheduler_concurrency=self._bf_concurrency,
                          query=meta(q),key=meta(k),value=meta(v),output=meta(o),table=meta(table),
                          starts=meta(starts),lens=meta(lens),options=options,physical_pages=pages,
                          parent_sha256=PARENT_SHA,arena_bytes=compact._arena['bytes'],performance_eligible=False)
            payload=dict(metadata=metadata,query=clone_layout(q,'cpu'),key=k.index_select(0,index).cpu(),
                         value=v.index_select(0,index).cpu(),expected=clone_layout(o,'cpu'),
                         table=clone_layout(table,'cpu'),starts=clone_layout(starts,'cpu'),lens=clone_layout(lens,'cpu'))
            path=self.run/('bf16-prefill-operands-'+'-'.join(map(str,key))+'.pt')
            torch.save(payload,path);metadata['file']=path.name;metadata['bytes']=path.stat().st_size
            self._bf_records.append(metadata);self._bf_done.add(key)
            (self.run/'bf16-prefill-capture.json').write_text(json.dumps(self._bf_records,indent=2)+'\n')
            return used
        compact.try_forward = capture
        (self.run/'bf16-prefill-binding.json').write_text(json.dumps(dict(mode=mode,parent_sha256=PARENT_SHA,
            delta='Native BF16 WMMA, explicit LDS K/V staging, standard moving-maximum softmax, true varlen/causal boundaries. Retained KV gather, decode, GDN, weights and DSpark unchanged.',
            sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [pathlib.Path(__file__),pathlib.Path(__file__).with_name('bf16_prefill_kernel.py'),pathlib.Path(__file__).with_name('bf16_prefill_native.py'),pathlib.Path(__file__).parent/'bf16-prefill-native-v1/prefill.hip',pathlib.Path(__file__).parent/'bf16-prefill-native-v1/prefill.so']},
            weights_changed=False,full_optimizer_qualified=False),indent=2)+'\n')
        return result
    def compile_or_warm_up_model(self):
        result=super().compile_or_warm_up_model()
        if os.environ.get('CIRU_BF16_PROFILE')=='1':
            from dspark_profile import Roctx
            self._bf_roctx=Roctx();self._bf_roctx.pause()
        self._bf_ready=True
        return result
    def execute_model(self,scheduler_output):
        self._bf_ordinal=0;self._bf_concurrency=len(scheduler_output.num_scheduled_tokens)
        return super().execute_model(scheduler_output)
    def _save_attention(self):
        super()._save_attention()
        if not getattr(self,'_bf_ready',False) or self._bf_stats is None:return
        values=self._bf_stats.cpu().tolist()
        (self.run/'bf16-prefill-audit.json').write_text(json.dumps(dict(columns=['unequal','nonfinite','max_abs','sum_squared_error','values','calls'],stats=values,scope='All real target prefill calls, identical live operands. Candidate served; synthetic warmup excluded. Timing invalid.'),indent=2)+'\n')
        assert values[1]==0,values
