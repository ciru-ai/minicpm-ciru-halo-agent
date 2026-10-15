"""C1-only packed draft head; the shared BF16 target module is untouched."""
import ctypes as C
import functools
import hashlib
import json
from pathlib import Path

import torch
from conv_grid_worker import Worker as Parent

HERE=Path(__file__).resolve().parent


class Worker(Parent):
    def load_model(self,*args,**kwargs):
        value=super().load_model(*args,**kwargs)
        self._dh_ready=False
        pins=json.loads((HERE/'sampling-dense/runtime/candidate-pins.json').read_text())
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
        manifest=json.loads((HERE/'sampling-dense/draft-head-q8-c1/manifest.json').read_text())
        packed=torch.load(HERE/'sampling-dense/draft-head-q8-c1-component/packed-draft-head.pt',map_location='cpu',weights_only=False)
        self._dh_codes=packed['codes'].to(self.device)
        self._dh_scales=packed['scales'].to(self.device)
        self._dh_flags=torch.zeros(1,dtype=torch.int32,device=self.device)
        self._dh_library=C.CDLL(manifest['library']);self._dh_launch=getattr(self._dh_library,manifest['symbol'])
        self._dh_launch.argtypes=[C.c_void_p]*6+[C.c_int]*3+[C.c_void_p];self._dh_launch.restype=C.c_int
        p=self.model_runner.speculator;draft=p.model;target=self.model_runner.get_model().language_model
        assert draft.lm_head is target.lm_head and draft.lm_head.weight.dtype==torch.bfloat16
        assert draft.logits_processor.scale==1 and draft.logits_processor.org_vocab_size==248144
        assert draft.logits_processor.head_dtype==torch.bfloat16
        assert p.draft_logits is None and p._draft_topk is None and draft.draft_id_to_target_id is None
        self._dh_shared_module=target.lm_head
        self._dh_shared_address=target.lm_head.weight.data_ptr()
        self._dh_native_compute=draft.compute_draft_logits
        self._dh_context=None
        self._dh_binding=dict(pins=pins,manifest=manifest,graphs={},
            target_draft_module_still_shared=True,target_head_weight_changed=False,
            extra_resident_packed_bytes=self._dh_codes.numel()+2*self._dh_scales.numel(),
            scope='Only draft compute_draft_logits for M3/depth3 uses Q8. Target, Markov, embedding, prefill and every other draft shape remain native BF16.',
            full_optimizer_qualified=False)

        @functools.wraps(self._dh_native_compute)
        def compute(hidden):
            selected=hidden.shape[0]==3 and p.num_speculative_steps==3
            context=self._dh_context
            if context is not None:
                context['calls'].append(dict(candidate=selected,shape=list(hidden.shape),stride=list(hidden.stride()),dtype=str(hidden.dtype)))
            if not selected:return self._dh_native_compute(hidden)
            assert tuple(hidden.shape)==(3,2048) and hidden.is_contiguous() and hidden.dtype==torch.bfloat16
            output=torch.empty((3,248192),dtype=hidden.dtype,device=hidden.device)
            status=self._dh_launch(hidden.data_ptr(),self._dh_codes.data_ptr(),self._dh_scales.data_ptr(),
                output.data_ptr(),self._dh_flags.data_ptr(),None,3,248192,2048,torch.cuda.current_stream().cuda_stream)
            assert status==0,status
            return output[:,:248144]
        draft.compute_draft_logits=compute
        original=p._generate_draft

        @functools.wraps(original)
        def generate(num_reqs,num_tokens_padded,*a,**kw):
            if not torch.cuda.is_current_stream_capturing():return original(num_reqs,num_tokens_padded,*a,**kw)
            key=[p.num_speculative_steps,num_reqs,num_tokens_padded]
            context=dict(key=key,calls=[])
            label=json.dumps(key);assert label not in self._dh_binding['graphs']
            self._dh_binding['graphs'][label]=context
            self._dh_context=context
            try:return original(num_reqs,num_tokens_padded,*a,**kw)
            finally:self._dh_context=None
        p._generate_draft=generate
        return value

    def compile_or_warm_up_model(self):
        value=super().compile_or_warm_up_model()
        graphs=self._dh_binding['graphs']
        for label,context in graphs.items():
            assert len(context['calls'])==1,(label,context)
            assert context['calls'][0]['candidate']==(context['key']==[3,1,3])
        assert graphs[json.dumps([3,1,3])]['calls'][0]['candidate']
        assert self.model_runner.get_model().language_model.lm_head is self._dh_shared_module
        assert self.model_runner.speculator.model.lm_head is self._dh_shared_module
        assert self._dh_shared_module.weight.data_ptr()==self._dh_shared_address
        assert int(self._dh_flags.item())==0
        self._dh_binding['warmup_flags']=0;self._dh_ready=True
        (self.run/'draft-head-binding.json').write_text(json.dumps(self._dh_binding,indent=2)+'\n')
        return value


from profile_worker import Worker as ProfilingParent
from dense_capture_worker import dense_observe_before,dense_observe_after


class ProfileWorker(ProfilingParent,Worker):
    def load_model(self,*args,**kwargs):
        value=super().load_model(*args,**kwargs)
        self._dh_audit={};self._dh_done=set();self._dh_rows=[]
        self._dh_markers={key:torch.empty(2,dtype=torch.int32,device=self.device) for key in ((3,1,3),(2,8,16))}
        self._dh_counts={key:torch.zeros(4,dtype=torch.int64,device=self.device) for key in self._dh_markers}
        p=self.model_runner.speculator;original=p._sample_sequential

        @functools.wraps(original)
        def sequential(num_reqs,head_hidden):
            value=original(num_reqs,head_hidden)
            context=self._dh_context
            key=tuple(context['key']) if context else None
            if key not in self._dh_markers:return value
            depth=key[0];num_sample=num_reqs*depth
            candidate=p.draft_tokens[:num_reqs,:depth].clone()
            hidden=head_hidden[p.sample_indices[:num_sample]]
            logits=self._dh_native_compute(hidden).view(num_reqs,depth,-1)
            prev=p.input_buffers.input_ids[p._anchor_idx[:num_reqs]]
            reference=[]
            for i in range(depth):
                bias=p.model.markov_bias(p.model.markov_embed(prev))
                prev=(logits[:,i]+bias).argmax(-1)
                reference.append(prev)
            native=torch.stack(reference,dim=1)
            matches=candidate==native
            counts=self._dh_counts[key]
            counts[0].add_(num_sample);counts[1].add_(matches.sum())
            counts[2].add_(num_reqs);counts[3].add_(matches.all(-1).sum())
            dense_observe_before[(1,)](self._dh_markers[key],0,num_warps=1)
            dense_observe_after[(1,)](self._dh_markers[key],0,num_warps=1)
            assert key not in self._dh_audit
            self._dh_audit[key]=dict(candidate=candidate,native=native,
                positions=p.sample_pos[:num_sample].clone(),mapping=p.sample_idx_mapping[:num_sample].clone())
            return value
        p._sample_sequential=sequential
        return value

    def compile_or_warm_up_model(self):
        for markers in self._dh_markers.values():
            dense_observe_before[(1,)](markers,0,num_warps=1)
            dense_observe_after[(1,)](markers,0,num_warps=1)
        value=super().compile_or_warm_up_model()
        assert set(self._dh_audit)==set(self._dh_markers)
        with torch.inference_mode():
            for counts in self._dh_counts.values():counts.zero_()
        return value

    def sample_tokens(self,*args,**kwargs):
        row=getattr(self,'_profile_pending',None)
        selected=getattr(self,'_dh_ready',False) and row and row['shape'] in ('c1-early','c1-late','c8-early','c8-late') and row['key'] not in self._dh_done
        if selected:
            key=(3,1,3) if row['requests']==1 else (2,8,16)
            with torch.inference_mode():self._dh_markers[key].zero_()
        value=super().sample_tokens(*args,**kwargs)
        if selected:
            assert self._dh_markers[key].cpu().tolist()==[1,1]
            assert int(self._dh_flags.item())==0
            data={k:v.cpu().tolist() for k,v in self._dh_audit[key].items()}
            if row['requests']==8:assert data['candidate']==data['native']
            self._dh_rows.append(dict(profile_index=row['index'],cohort=row['cohort'],shape=row['shape'],
                requests=row['requests'],draft_dispatch=dict(self.model_runner.speculator._depth_last),data=data,
                cumulative_counts={str(k):v.cpu().tolist() for k,v in self._dh_counts.items()}))
            self._dh_done.add(row['key'])
            (self.run/'draft-head-live-audit.json').write_text(json.dumps(dict(flags=0,rows=self._dh_rows,
                counters_order=['proposals','matching_proposals','sequences','matching_sequences'],
                scope='Each selected draft graph also runs an independent original-BF16 full sequential Markov reference on identical live hidden states and anchors. Native reference does not write drafter state. Counters exclude startup and cover only the two admitted graphs.',
                performance_eligible=False),indent=2)+'\n')
        return value
