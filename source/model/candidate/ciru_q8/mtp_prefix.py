"""Check/annotate the single native Apodex MTP cache group for prefix reuse."""
_original=None
def install():
    global _original
    from vllm.v1.core import kv_cache_utils
    if _original is not None:return
    _original=kv_cache_utils.get_kv_cache_groups
    def groups(config,specs):
        result=_original(config,specs);s=config.speculative_config
        if config.model_config.quantization!='ciru_q8' or s is None or s.method not in ('mtp','dspark'):return result
        from vllm.v1.kv_cache_interface import FullAttentionSpec,MambaSpec
        targets={n for n in specs if n.startswith('language_model.model.layers.')}
        draft=set(specs)-targets
        expected={'mtp.layers.0.self_attn.attn'} if s.method=='mtp' else {f'model.layers.{i}.self_attn.attn' for i in range(40,45)}
        assert len(targets)==40 and draft==expected,sorted(specs)
        assert sum(isinstance(specs[n],MambaSpec) for n in targets)==30
        selected=[g for g in result if next(iter(draft)) in g.layer_names]
        assert len(selected)==1 and isinstance(selected[0].kv_cache_spec,FullAttentionSpec)
        assert draft.issubset(set(selected[0].layer_names))
        # The native planner may put equal-spec target and MTP full-attention
        # layers in one group. Mark that full-attention group as in the retained
        # native-MTP prefix repair; never widen any Mamba group's lookup.
        assert not any(g.is_eagle_group for g in result if g is not selected[0])
        if not s.use_eagle_block_drop():
            assert not any(g.is_eagle_group for g in result)
            print('CIRU_Q8_DSPARK_UNSHIFTED_PREFIX native trailing-block reuse enabled',flush=True)
            return result
        selected[0].is_eagle_group=True
        print('CIRU_Q8_NATIVE_MTP_PREFIX_GROUPS',[(g.layer_names,g.is_eagle_group) for g in result],flush=True)
        return result
    kv_cache_utils.get_kv_cache_groups=groups
