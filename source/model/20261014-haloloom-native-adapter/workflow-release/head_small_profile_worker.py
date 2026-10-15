"""Existing selected profiler narrowed to actual C2/C3 draft captures."""
import collections,json,os,time
from pathlib import Path
import torch
from profile_worker import Worker as ProfilingParent
from draft_head_worker import Worker as DraftHeadParent

class Worker(ProfilingParent, DraftHeadParent):
    def _save_profile(self):
        required={'amd-c3/c3-early','amd-c3/c3-late','amd-c2/c2-early','amd-c2/c2-late'}
        seen={r['key'] for r in self._profile_rows}
        if not self._profile_detached and required<=seen:
            assert self._profile_pending is None
            torch.cuda.synchronize(self.device)
            self._profile_roctx.detach();self._profile_detached=True
            (self.run/'profile-detach.json').write_text(json.dumps(dict(status='EXPORTED_BEFORE_WORKER_EXIT',required=sorted(required),seen=sorted(seen)))+'\n')
        (self.run/'live-profile.json').write_text(json.dumps(dict(steps=self._profile_rows,histogram=dict(self._profile_histogram),detached=self._profile_detached,diagnostic_only=True,performance_eligible=False,quality_evidence=False,full_optimizer_qualified=False),indent=2)+'\n')

    def execute_model(self, scheduler_output, *args, **kwargs):
            if not hasattr(self, '_profile_roctx') or self._profile_detached:
                return super(ProfilingParent, self).execute_model(scheduler_output, *args, **kwargs)
            if scheduler_output.scheduled_new_reqs:
                self._profile_control = json.loads(Path(os.environ['CIRU_DEPTH_CONTROL']).read_text())
            label = self._profile_control['label']
            counts = scheduler_output.num_scheduled_tokens
            drafts = scheduler_output.scheduled_spec_decode_tokens
            total = sum(counts.values())
            concurrency = len(counts)
            cached = scheduler_output.scheduled_cached_reqs
            outputs = list(map(int, cached.num_output_tokens))
            shape = None
            if counts and max(counts.values()) > 4:
                shape = None
            elif concurrency in (2, 3) and all(n == 3
                    and len(drafts.get(k, [])) == n - 1 for k, n in counts.items()):
                if outputs and min(outputs) >= 128:
                    shape = f'c{concurrency}-late'
                elif outputs and min(outputs) >= 8:
                    shape = f'c{concurrency}-early'
            key = label + '/' + str(shape)
            self._profile_histogram[key] += 1
            seen = sum(r['key'] == key for r in self._profile_rows)
            limit = 1
            if shape is None or seen >= limit or not self._profile_control.get('capture_head', True):
                return super(ProfilingParent, self).execute_model(scheduler_output, *args, **kwargs)
            if shape == 'prefill' and self._profile_skipped[key] < 2:
                self._profile_skipped[key] += 1
                return super(ProfilingParent, self).execute_model(scheduler_output, *args, **kwargs)
            assert self._profile_pending is None
            row = dict(key=key, cohort=label, shape=shape, index=len(self._profile_rows),
                scheduled_tokens=total, requests=concurrency, host_stages=[], worker_pid=os.getpid(),
                scheduled_counts=dict(counts), draft_counts={k: len(v) for k, v in drafts.items()},
                cached_computed=list(map(int, cached.num_computed_tokens)), cached_output=outputs,
                next_depth=scheduler_output.num_spec_tokens_to_schedule)
            self._profile_pending = row
            marker = f'M60_{row["index"]}_TARGET'
            row['target_marker'] = marker
            torch.cuda.synchronize(self.device)
            self._profile_roctx.resume()
            depth = self._profile_roctx.push(marker)
            start = time.monotonic_ns()
            try:
                value = super(ProfilingParent, self).execute_model(scheduler_output, *args, **kwargs)
                torch.cuda.synchronize(self.device)
            finally:
                self._profile_roctx.pop(depth)
                self._profile_roctx.pause()
            row['target_wall_ms'] = (time.monotonic_ns()-start)/1e6
            if value is not None:
                self._profile_rows.append(row)
                self._profile_pending = None
                self._save_profile()
            return value
