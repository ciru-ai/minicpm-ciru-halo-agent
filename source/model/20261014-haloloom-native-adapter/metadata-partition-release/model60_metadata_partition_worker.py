"""Avoid sorting an already partitioned speculative-prefix metadata batch."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap

import torch
from model60_small_prefill_worker import Worker as Parent


class Worker(Parent):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder as Builder
        original = Builder.build
        source_path = Path(inspect.getfile(original))
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert digest == 'ce951db92bb58b5b655ff00a89d0ffc68f8fbfb936e1e2a72a5da543cdf9d2f4', (str(source_path), digest)
        source = textwrap.dedent(inspect.getsource(original))
        start = source.index('            spec_token_masks = torch.repeat_interleave(')
        stop = source.index('            spec_state_indices_tensor = block_table_tensor[', start)
        replaced = source[start:stop]
        assert 'torch.argsort(spec_token_masks, stable=True)' in replaced
        assert 'spec_token_indx = index[num_non_spec_tokens:]' in replaced
        replacement = '''            num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
            non_spec_token_indx, spec_token_indx = _model60_mixed_partition(
                spec_sequence_masks, spec_sequence_masks_cpu, query_lens,
                query_start_loc_cpu, num_spec_decodes, num_non_spec_tokens)

'''
        source = source[:start] + replacement + source[stop:]
        audit = os.environ.get('CIRU_METADATA_PARTITION_AUDIT') == '1'
        seen = set()
        observations = []
        binding = dict(source=str(source_path), source_sha256=digest,
            worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            replaced_block_sha256=hashlib.sha256(replaced.encode()).hexdigest(),
            transformed_build_sha256=hashlib.sha256(source.encode()).hexdigest(),
            audit=audit, target_math_unchanged=True, old_sort_preserved_for_nonprefix=True,
            full_optimizer_qualified=False, calls=0, prefix_calls=0, fallback_calls=0,
            observations=observations)

        def save():
            (self.run/'metadata-partition-binding.json').write_text(json.dumps(binding, indent=2)+'\n')

        def partition(mask, mask_cpu, lengths, starts_cpu, spec_reqs, nonspec_tokens):
            total = int(starts_cpu[-1])
            spec_tokens = total - nonspec_tokens
            cpu_mask = mask_cpu.numpy()
            prefix = bool(cpu_mask[:spec_reqs].all()) and not bool(cpu_mask[spec_reqs:].any())
            binding['calls'] += 1
            binding['prefix_calls' if prefix else 'fallback_calls'] += 1
            if prefix:
                assert int(starts_cpu[spec_reqs]) == spec_tokens
                index = torch.arange(total, dtype=torch.int64, device=lengths.device)
                non_spec = index[spec_tokens:]
                spec = index[:spec_tokens]
            else:
                token_mask = torch.repeat_interleave(mask, lengths, output_size=total)
                index = torch.argsort(token_mask, stable=True)
                non_spec, spec = index[:nonspec_tokens], index[nonspec_tokens:]
            key = (tuple(starts_cpu.tolist()), tuple(mask_cpu.tolist()))
            if audit or (key not in seen and len(seen) < 32):
                row = dict(query_starts=list(key[0]), spec_mask=list(key[1]),
                    total=total, spec_tokens=spec_tokens, prefix=prefix,
                    device=str(lengths.device), lengths_dtype=str(lengths.dtype),
                    index_dtype=str(index.dtype), index_stride=list(index.stride()))
                if audit:
                    reference_mask = torch.repeat_interleave(mask, lengths, output_size=total)
                    reference = torch.argsort(reference_mask, stable=True)
                    row['non_spec_exact'] = torch.equal(non_spec, reference[:nonspec_tokens])
                    row['spec_exact'] = torch.equal(spec, reference[nonspec_tokens:])
                    assert row['non_spec_exact'] and row['spec_exact'], row
                    # Retain actual operand and full consumed-index identity.
                    raw = torch.cat((non_spec, spec)).cpu().numpy().tobytes()
                    row['consumed_index_sha256'] = hashlib.sha256(raw).hexdigest()
                observations.append(row)
                seen.add(key)
                save()
            return non_spec, spec

        namespace = dict(original.__globals__, _model60_mixed_partition=partition)
        exec(compile(source, str(Path(__file__)) + '.build', 'exec'), namespace)
        Builder.build = functools.update_wrapper(namespace['build'], original)
        self._metadata_partition_save = save
        self._metadata_partition_binding = binding
        save()
        return result

    def shutdown(self, *args, **kwargs):
        if hasattr(self, '_metadata_partition_save'):
            self._metadata_partition_save()
        return super().shutdown(*args, **kwargs)
