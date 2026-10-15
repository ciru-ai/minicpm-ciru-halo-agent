"""Target-only fixed tile clone of the retained AMD BF16 prefill wrapper."""
import hashlib
import inspect
import pathlib
import types

PARENT_SHA = '349638519a32b3ff477b630bcbd1d9b6a1957b137967cde0e7f61b8131425334'

class FixedTile:
    def __init__(self, parent, block_m, **overrides):
        assert block_m in (64, 128)
        self.kernel = parent.fn
        assert type(self.kernel).__name__ == 'JITFunction'
        config = parent.configs
        assert len(config) == 1
        config = config[0]
        assert config.kwargs == dict(BLOCK_M=128, BLOCK_N=32, PRE_LOAD_V=False, waves_per_eu=6), config.kwargs
        assert config.num_warps == 8 and config.num_stages == 1
        self.options = {**config.kwargs, 'BLOCK_M': block_m,
                        'num_warps': config.num_warps, 'num_stages': config.num_stages}
        assert set(overrides) <= {"num_warps", "waves_per_eu", "matrix_instr_nonkdim", "kpack"}
        self.options.update(overrides)
        self.last = None
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            assert not (self.options.keys() & kwargs.keys())
            result = self.kernel[grid](*args, **kwargs, **self.options)
            self.last = result
            return result
        return launch
    def receipt(self):
        kernel = self.last
        if kernel is None:
            return dict(options=self.options, compiled=False)
        metadata = kernel.metadata
        return dict(options=self.options, compiled=True, n_regs=getattr(kernel, 'n_regs', None),
                    n_spills=getattr(kernel, 'n_spills', None), hash=getattr(kernel, 'hash', None),
                    metadata=metadata._asdict() if hasattr(metadata, '_asdict') else str(metadata))

def clone_prefill(parent, block_m, **overrides):
    from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import fwd_prefill
    path = pathlib.Path(inspect.getfile(fwd_prefill))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PARENT_SHA
    fixed = FixedTile(parent.__globals__['attn_fwd'], block_m, **overrides)
    fn = types.FunctionType(parent.__code__, {**parent.__globals__, 'attn_fwd': fixed},
                            parent.__name__, parent.__defaults__, parent.__closure__)
    fn.__kwdefaults__ = parent.__kwdefaults__
    return fn, fixed
