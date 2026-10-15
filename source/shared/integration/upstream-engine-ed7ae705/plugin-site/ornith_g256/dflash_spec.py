"""Keep DFlash query slots after the runner supplies target context metadata."""
# Copyright 2026 Ciru.
from vllm.v1.spec_decode.dflash import DFlashProposer


_upstream_set_inputs_first_pass = DFlashProposer.set_inputs_first_pass


def set_inputs_first_pass(self, *args, **kwargs):
    # Upstream now forwards query_metadata.slot_mapping directly into its
    # single draft-group buffer. The old per-group source override is gone.
    # Preserve the one-group restriction and verify the returned query view.
    if any(group.kv_cache_group_id != self.kv_cache_gid
           for group in self.draft_attn_groups):
        raise ValueError('Ornith DFlash requires one draft KV cache group')
    result = _upstream_set_inputs_first_pass(self, *args, **kwargs)
    _, _, query_metadata = result
    assert query_metadata.slot_mapping.data_ptr() == self._slot_mapping_buffer.data_ptr()
    assert query_metadata.slot_mapping.numel() == result[0]
    return result


def install():
    """Patch only DFlash's input hook; leave installed vLLM files untouched."""
    current = DFlashProposer.set_inputs_first_pass
    if current not in (_upstream_set_inputs_first_pass, set_inputs_first_pass):
        raise RuntimeError('Another extension replaced DFlash input preparation')
    DFlashProposer.set_inputs_first_pass = set_inputs_first_pass
