"""Add the qualified M16 draft projection while retaining the frozen C1 winner."""
import ctypes as C
import functools
import hashlib
import json
from pathlib import Path

import torch
from draft_head_worker import Worker as Parent
from draft_head_worker import ProfileWorker as AuditParent

HERE=Path(__file__).resolve().parent


class Worker(Parent):
    def load_model(self,*args,**kwargs):
        value=super().load_model(*args,**kwargs)
        candidate=HERE/'sampling-dense/draft-head-q8-c8-wmma-packed-v1'
        manifest=json.loads((candidate/'manifest.json').read_text())
        assert hashlib.sha256(Path(manifest['library']).read_bytes()).hexdigest()==manifest['library_sha256']
        assert hashlib.sha256((candidate/'candidate-source.hip').read_bytes()).hexdigest()==manifest['source_sha256']
        self._c8_codes=self._dh_codes.view(248192//16,16,64,2,16).permute(0,2,3,1,4).contiguous()
        self._c8_scales=self._dh_scales.view(248192//16,16,64).permute(0,2,1).contiguous()
        assert torch.equal(self._c8_codes.permute(0,3,1,2,4).reshape_as(self._dh_codes),self._dh_codes)
        assert torch.equal(self._c8_scales.permute(0,2,1).reshape_as(self._dh_scales),self._dh_scales)
        self._c8_library=C.CDLL(manifest['library'])
        self._c8_launch=getattr(self._c8_library,manifest['symbol'])
        self._c8_launch.argtypes=[C.c_void_p]*6+[C.c_int]*3+[C.c_void_p];self._c8_launch.restype=C.c_int
        self._c8_binding=dict(manifest=manifest,worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            parent_file=str(Path(__import__('inspect').getfile(Parent))),
            layout_values_bitwise_preserved=True,additional_resident_bytes=self._c8_codes.numel()+2*self._c8_scales.numel(),
            shared_bf16_target_unchanged=True,c1_kernel_and_layout_unchanged=True,
            scope='Only captured M16/depth2 draft head changes. This graph serves actual C8 and padded C7. Other shapes retain the frozen parent.',
            full_optimizer_qualified=False)
        p=self.model_runner.speculator;draft=p.model;original=draft.compute_draft_logits
        @functools.wraps(original)
        def compute(hidden):
            if hidden.shape[0]!=16 or p.num_speculative_steps!=2:return original(hidden)
            assert tuple(hidden.shape)==(16,2048) and hidden.is_contiguous() and hidden.dtype==torch.bfloat16
            context=self._dh_context
            if context is not None:
                # Parent assertions describe its C1 branch; record the separate C8 route explicitly.
                context['calls'].append(dict(candidate=False,c8_candidate=True,shape=list(hidden.shape),stride=list(hidden.stride()),dtype=str(hidden.dtype)))
            output=torch.empty((16,248192),dtype=hidden.dtype,device=hidden.device)
            status=self._c8_launch(hidden.data_ptr(),self._c8_codes.data_ptr(),self._c8_scales.data_ptr(),
                output.data_ptr(),self._dh_flags.data_ptr(),None,16,248192,2048,torch.cuda.current_stream().cuda_stream)
            assert status==0,status
            return output[:,:248144]
        draft.compute_draft_logits=compute
        return value

    def compile_or_warm_up_model(self):
        value=super().compile_or_warm_up_model()
        graphs=self._dh_binding['graphs']
        selected=[key for key,ctx in graphs.items() if any(c.get('c8_candidate') for c in ctx['calls'])]
        assert selected==[json.dumps([2,8,16])],selected
        assert int(self._dh_flags.item())==0
        self._c8_binding.update(graphs=graphs,warmup_flags=0,allocated_bytes=torch.cuda.memory_allocated(self.device))
        (self.run/'draft-head-c8-binding.json').write_text(json.dumps(self._c8_binding,indent=2)+'\n')
        return value


class ProfileWorker(AuditParent,Worker):
    def sample_tokens(self,*args,**kwargs):
        row=getattr(self,'_profile_pending',None)
        selected=getattr(self,'_dh_ready',False) and row and row['shape'] in ('c1-early','c1-late','c8-early','c8-late') and row['key'] not in self._dh_done
        if selected:
            key=(3,1,3) if row['requests']==1 else (2,8,16)
            with torch.inference_mode():self._dh_markers[key].zero_()
        # Retain the independent BF16 audit, but do not assume the new C8 quantized proposals are exact.
        value=super(AuditParent,self).sample_tokens(*args,**kwargs)
        if selected:
            assert self._dh_markers[key].cpu().tolist()==[1,1]
            assert int(self._dh_flags.item())==0
            data={k:v.cpu().tolist() for k,v in self._dh_audit[key].items()}
            self._dh_rows.append(dict(profile_index=row['index'],cohort=row['cohort'],shape=row['shape'],
                requests=row['requests'],draft_dispatch=dict(self.model_runner.speculator._depth_last),data=data,
                cumulative_counts={str(k):v.cpu().tolist() for k,v in self._dh_counts.items()}))
            self._dh_done.add(row['key'])
            (self.run/'draft-head-c8-live-audit.json').write_text(json.dumps(dict(flags=0,rows=self._dh_rows,
                counters_order=['proposals','matching_proposals','sequences','matching_sequences'],
                scope='Independent original-BF16 full sequential Markov reference on identical live hidden states and anchors. Selected rows are actual C1/C8. Cumulative counters count captured graph rows and may include C7 padding; they are not a count of useful real-request proposals.',
                performance_eligible=False),indent=2)+'\n')
        return value
