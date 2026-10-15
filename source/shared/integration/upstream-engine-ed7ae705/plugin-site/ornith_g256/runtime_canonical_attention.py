"""Preserve released single-query attention partitions in speculative decode."""
def install_worker_hook(worker):
    # Declare ownership before worker construction. The parent otherwise
    # installs a persistent forward plus cache-write/lifecycle hooks; this
    # canonical forward later supersedes only the read side of that bundle.
    worker._ornith_attention_policy = 'canonical_partitioned'
    original=worker.init_device
    def initialize(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        from . import attention_folded,attention_canonical_folded
        attention_folded.forward=attention_canonical_folded.forward
        print('ORNITH_CANONICAL_PARTITION_KV_SHARING tile4 boundary_fallback',flush=True)
        return result
    worker.init_device=initialize
