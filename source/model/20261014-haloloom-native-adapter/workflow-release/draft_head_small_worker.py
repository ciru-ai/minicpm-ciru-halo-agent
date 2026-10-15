"""Extend the retained packed draft head to qualified actual M4/M6 shapes."""
import functools
import hashlib
import inspect
import json
from pathlib import Path
import torch
from draft_head_c8_worker import Worker as C8Parent

class Worker(C8Parent):
    def load_model(self,*args,**kwargs):
        value=super().load_model(*args,**kwargs)
        self._small_live={m:torch.tensor([m],dtype=torch.int32,device=self.device) for m in (4,6)}
        draft=self.model_runner.speculator.model
        p=self.model_runner.speculator
        original=draft.compute_draft_logits
        @functools.wraps(original)
        def compute(hidden):
            m=hidden.shape[0]
            if m not in self._small_live or p.num_speculative_steps!=2:return original(hidden)
            assert tuple(hidden.shape)==(m,2048) and hidden.is_contiguous() and hidden.dtype==torch.bfloat16
            context=self._dh_context
            if context is not None:
                context['calls'].append(dict(candidate=False,c8_candidate=False,small_candidate=True,
                    shape=list(hidden.shape),stride=list(hidden.stride()),dtype=str(hidden.dtype),kernel_rows=16,live_rows=m))
            output=torch.empty((16,248192),dtype=hidden.dtype,device=hidden.device)
            status=self._c8_launch(hidden.data_ptr(),self._c8_codes.data_ptr(),self._c8_scales.data_ptr(),
                output.data_ptr(),self._dh_flags.data_ptr(),self._small_live[m].data_ptr(),
                16,248192,2048,torch.cuda.current_stream().cuda_stream)
            assert status==0,status
            return output[:m,:248144]
        draft.compute_draft_logits=compute
        return value

    def compile_or_warm_up_model(self):
        value=super().compile_or_warm_up_model()
        graphs=self._dh_binding['graphs']
        selected=[key for key,context in graphs.items() if any(c.get('small_candidate') for c in context['calls'])]
        assert set(selected)=={json.dumps([2,2,4]),json.dumps([2,3,6])},selected
        assert int(self._dh_flags.item())==0
        binding=dict(worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        binding.update(parent_worker_sha256=hashlib.sha256(Path(inspect.getfile(C8Parent)).read_bytes()).hexdigest(),
            library=self._c8_binding['manifest']['library'],library_sha256=self._c8_binding['manifest']['library_sha256'],
            extra_resident_bytes_vs_c8_candidate=8,extra_packed_matrix=False,
            shared_bf16_target_unchanged=True,c1_kernel_and_layout_unchanged=True,c8_candidate_unchanged=True,
            small_graphs=selected,graphs=graphs,warmup_flags=0,allocated_bytes=torch.cuda.memory_allocated(self.device),
            scope='M4/C2 and M6/C3 use the existing packed WMMA with live_rows; original C1 and packed M16 C7/C8 stay unchanged. C4-C6 retain native BF16.',
            full_optimizer_qualified=False)
        (self.run/'draft-head-small-binding.json').write_text(json.dumps(binding,indent=2)+'\n')
        return value


from head_small_profile_worker import Worker as SmallProfilingParent
from dense_capture_worker import dense_observe_before,dense_observe_after

class ProfileWorker(SmallProfilingParent,Worker):
    def load_model(self,*args,**kwargs):
        value=super().load_model(*args,**kwargs)
        self._dh_audit={};self._dh_done=set();self._dh_rows=[]
        self._dh_markers={key:torch.empty(2,dtype=torch.int32,device=self.device) for key in ((2,2,4),(2,3,6))}
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
        selected=getattr(self,'_dh_ready',False) and row and row['shape'] in ('c2-early','c2-late','c3-early','c3-late') and row['key'] not in self._dh_done
        if selected:
            key=(2,row['requests'],2*row['requests'])
            with torch.inference_mode():self._dh_markers[key].zero_()
        value=super().sample_tokens(*args,**kwargs)
        if selected:
            assert self._dh_markers[key].cpu().tolist()==[1,1]
            assert int(self._dh_flags.item())==0
            data={k:v.cpu().tolist() for k,v in self._dh_audit[key].items()}
            self._dh_rows.append(dict(profile_index=row['index'],cohort=row['cohort'],shape=row['shape'],
                requests=row['requests'],draft_dispatch=dict(self.model_runner.speculator._depth_last),data=data,
                cumulative_counts={str(k):v.cpu().tolist() for k,v in self._dh_counts.items()}))
            self._dh_done.add(row['key'])
            (self.run/'draft-head-small-live-audit.json').write_text(json.dumps(dict(flags=0,rows=self._dh_rows,
                counters_order=['proposals','matching_proposals','sequences','matching_sequences'],
                scope='Independent BF16 full sequential Markov reference on identical live hidden states and anchors; no drafter-state writes. Selected rows prove actual C2/C3 concurrency. Cumulative counters count graph rows and are not useful-request acceptance; startup excluded.',
                performance_eligible=False),indent=2)+'\n')
        return value
