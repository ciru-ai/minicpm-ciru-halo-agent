"""Existing selected-region profiler around the frozen draft-head release."""
from profile_worker import Worker as ProfilingParent
from draft_head_worker import Worker as DraftHeadParent


class Worker(ProfilingParent, DraftHeadParent):
    pass
