"""Existing selected-region profiler around the retained convolution child."""
from profile_worker import Worker as ProfilingParent
from conv_grid_worker import Worker as ConvolutionParent


class Worker(ProfilingParent, ConvolutionParent):
    pass
