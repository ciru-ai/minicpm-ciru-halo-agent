"""Own isolated MiniCPM DSpark and V2 AR with retained target error lifecycle."""
import json
from types import MethodType
from vllm import ir

def draft_norm(self,x,residual=None):
    if residual is None:
        return ir.ops.rms_norm.impls["native"].impl_fn(x,self.weight.data if self.pass_weight else None,self.variance_epsilon,self.variance_size_override)
    return ir.ops.fused_add_rms_norm.impls["native"].impl_fn(x,residual,self.weight.data if self.pass_weight_add else None,self.variance_epsilon,self.variance_size_override)

from vllm.v1.worker.gpu_worker import Worker as VllmGPUWorker
from .worker import Worker as TargetWorker

class Worker(TargetWorker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        c=self.vllm_config;s=c.speculative_config
        assert c.model_config.architecture=='MiniCPMV4_7ForConditionalGeneration'
        assert s is None or s.method=='dspark'
        if s is not None:
            from ornith_g256.gdn_spec import install
            install()
    def load_model(self,*args,**kwargs):
        result=super().load_model(*args,**kwargs)
        runner=self.model_runner
        s=self.vllm_config.speculative_config
        receipt=dict(runner_class=type(runner).__module__,speculative=s is not None,
                     target_native_error_snapshot='After execute_model and sample_tokens including drafting',qualified=False)
        if s is not None:
            proposer=runner.speculator;draft=proposer.model
            from vllm.model_executor.layers.layernorm import RMSNorm
            norms=[]
            for name,m in draft.named_modules():
                if type(m) is RMSNorm:
                    m.forward=MethodType(draft_norm,m);norms.append(name)
            receipt["draft_native_norms"]=norms
            import os,torch
            if os.environ.get('CIRU_DSPARK_TRACE')=='1':
                original_propose=proposer.propose;trace_count=[0]
                def traced_propose(input_batch,*args,**kwargs):
                    batch=input_batch
                    result=original_propose(batch,*args,**kwargs)
                    length=int(batch.seq_lens_cpu_upper_bound[:batch.num_reqs].max())
                    if length>60000 and trace_count[0]<10 and not kwargs.get('dummy_run',False):
                        trace_count[0]+=1
                        idx=batch.idx_mapping
                        row=dict(index=trace_count[0],sequence_upper_bound=length,requests=batch.req_ids,
                            num_target_tokens=batch.num_tokens,
                            target_input_ids=batch.input_ids.cpu().tolist(),
                            target_positions=batch.positions.cpu().tolist(),
                            context_positions=proposer.context_positions[:batch.num_tokens].cpu().tolist(),
                            context_slots=proposer._context_slot_mappings[:,:batch.num_tokens].cpu().tolist(),
                            last_sampled=runner.req_states.last_sampled_tokens[idx].cpu().tolist(),
                            draft_tokens=result.cpu().tolist(),
                            query_input_ids=proposer.input_buffers.input_ids[:batch.num_reqs*proposer.num_query_per_req].cpu().tolist(),
                            query_positions=proposer.input_buffers.positions[:batch.num_reqs*proposer.num_query_per_req].cpu().tolist())
                        with (self.run/'long-draft-trace.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    return result
                proposer.propose=traced_propose
            if os.environ.get('CIRU_DSPARK_KV_AUDIT')=='1':
                from .dspark_kv_audit import install as install_kv_audit
                install_kv_audit(self,proposer)
            from .dspark_attention_bridge import install
            install(self,self.vllm_config.additional_config["ciru_q8"].get("draft_attention","retained"))
            from .method import CiruQ8MoEMethod
            assert not any(isinstance(getattr(m,'quant_method',None),CiruQ8MoEMethod) for m in draft.modules())
            assert type(proposer).__name__=='DSparkSpeculator'
            receipt.update(method=s.method,drafts=s.num_speculative_tokens,draft_sample_method=s.draft_sample_method,
                model_class=type(draft).__name__,draft_parameters=sum(p.numel() for p in draft.parameters()),
                sample_from_anchor=proposer.sample_from_anchor,query_per_request=proposer.num_query_per_req,
                target_layers=list(runner.get_model().get_language_model().model.aux_hidden_state_layers),
                confidence_head=draft.model.confidence_head is not None,
                drafter_model=s.model,enable_adaptive_verification=s.enable_adaptive_verification)
        (self.run/'dspark-binding.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return result
    def compile_or_warm_up_model(self):
        self._dspark_warmup=True
        try:
            result=super().compile_or_warm_up_model()
            runner=self.model_runner
            if self.vllm_config.speculative_config is not None:
                (self.run/'dspark-cache-binding.json').write_text(json.dumps(dict(
                    draft_group_ids=runner.speculator.draft_kv_cache_group_ids,
                    groups=[dict(layers=g.layer_names,is_eagle_group=g.is_eagle_group,spec=str(g.kv_cache_spec)) for g in runner.kv_cache_config.kv_cache_groups],
                    sampler=type(runner.rejection_sampler).__module__,
                    warmup_error_scope='One outer checked lifecycle region; V2 synthetic step outputs are not consumed'),indent=2)+'\n')
            if getattr(self,"_draft_attention_stats",None) is not None:self._draft_attention_stats.zero_()
            return result
        finally:self._dspark_warmup=False
    def _save_attention(self):
        super()._save_attention()
        stats=getattr(self,'_draft_attention_stats',None)
        if stats is not None and not getattr(self,'_dspark_warmup',False):
            v=stats.cpu().tolist()
            (self.run/'draft-attention-actual-operands.json').write_text(json.dumps(dict(real_dispatches=v[0],max_abs=v[1],max_dispatch_rms=v[2],nonfinite=v[3],compared_values=v[4],scope='Both operators use identical live Q/K/V/cache metadata, including captured graph replay. Padding excluded. Synthetic statistics reset.'),indent=2)+'\n')
    def execute_model(self,scheduler_output):
        if getattr(self,'_dspark_warmup',False):return VllmGPUWorker.execute_model(self,scheduler_output)
        lifecycle=self._require_lifecycle()
        if self._ornith_pending is not None:raise RuntimeError('Previous target/draft step unfinished')
        ticket=lifecycle.begin('execute-target-and-dspark')
        try:output=VllmGPUWorker.execute_model(self,scheduler_output)
        except Exception as error:lifecycle.abort(ticket,error)
        if output is None:self._ornith_pending=ticket;return None
        lifecycle.queue(ticket);result=lifecycle.checked_output(output,ticket);self._save_attention();return result
    def sample_tokens(self,grammar_output):
        if getattr(self,'_dspark_warmup',False):return VllmGPUWorker.sample_tokens(self,grammar_output)
        lifecycle=self._require_lifecycle();ticket=self._ornith_pending
        if ticket is None:return VllmGPUWorker.sample_tokens(self,grammar_output)
        self._ornith_pending=None
        try:output=VllmGPUWorker.sample_tokens(self,grammar_output)
        except Exception as error:lifecycle.abort(ticket,error)
        lifecycle.queue(ticket);result=lifecycle.checked_output(output,ticket);self._save_attention();return result
