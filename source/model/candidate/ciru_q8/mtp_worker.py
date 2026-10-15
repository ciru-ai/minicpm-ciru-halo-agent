"""Own native BF16 Apodex MTP with the retained target error lifecycle."""
import json
from vllm.v1.worker.gpu_worker import Worker as VllmGPUWorker
from .worker import Worker as TargetWorker

class Worker(TargetWorker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        c=self.vllm_config;s=c.speculative_config
        assert s is not None and s.method=='mtp' and s.num_speculative_tokens in (1,2,4,8)
        assert c.model_config.architecture=='Qwen3_5MoeForConditionalGeneration'
        assert c.model_config.hf_text_config.vocab_size==248320
        from ornith_g256.gdn_spec import install
        install()
    def load_model(self,*args,**kwargs):
        result=super().load_model(*args,**kwargs)
        draft=self.model_runner.drafter.model
        from .method import CiruQ8MoEMethod
        assert not any(isinstance(getattr(m,'quant_method',None),CiruQ8MoEMethod) for m in draft.modules())
        (self.run/'mtp-binding.json').write_text(json.dumps(dict(
            method='mtp',drafts=self.vllm_config.speculative_config.num_speculative_tokens,
            model_class=type(draft).__name__,draft_parameters=sum(p.numel() for p in draft.parameters()),
            checkpoint=self.vllm_config.model_config.model,own_source_bf16_mtp=True,
            target_native_error_snapshot='After execute_model and sample_tokens including draft generation',
            legacy_slot_patch='Not installed: pinned proposer has no stale per-group slot map; serial steps use its owned slot buffer',
            qualified=False),indent=2)+'\n')
        return result
    def execute_model(self,scheduler_output):
        lifecycle=self._require_lifecycle()
        if self._ornith_pending is not None:raise RuntimeError('Previous target/draft step unfinished')
        ticket=lifecycle.begin('execute-target-and-native-mtp')
        try:output=VllmGPUWorker.execute_model(self,scheduler_output)
        except Exception as error:lifecycle.abort(ticket,error)
        if output is None:self._ornith_pending=ticket;return None
        lifecycle.queue(ticket);result=lifecycle.checked_output(output,ticket);self._save_attention();return result
    def sample_tokens(self,grammar_output):
        lifecycle=self._require_lifecycle();ticket=self._ornith_pending
        if ticket is None:return VllmGPUWorker.sample_tokens(self,grammar_output)
        self._ornith_pending=None
        try:output=VllmGPUWorker.sample_tokens(self,grammar_output)
        except Exception as error:lifecycle.abort(ticket,error)
        lifecycle.queue(ticket);result=lifecycle.checked_output(output,ticket);self._save_attention();return result
