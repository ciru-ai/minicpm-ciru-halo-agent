# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ciru. Generated arithmetic retains the installed vLLM/SGLang
# Apache-2.0 decoder. See attention_partition.py for the original credits.
"""Share KV within four-query tiles without changing canonical partitions.

The canonical split width is ceil(causal_query_length / 32). Four adjacent
queries can share KV iff that width is identical. A boundary-crossing tile
uses the unmodified per-query arithmetic instead. The common reducer always
uses per-query causal lengths, including graph-padded rows.
"""
from functools import lru_cache
import inspect
import linecache

import torch
from vllm.triton_utils import triton
from . import attention_partition as canonical


def _compile(source, namespace, suffix):
    filename = __file__ + suffix
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, 'exec'), namespace)


def _replace(source, replacements):
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError('Unsupported canonical-folded source anchor: ' + old[:80])
        source = source.replace(old, new)
    return source


@lru_cache(maxsize=4)
def _kernels(query_tiles):
    native, reduce = canonical._kernels()
    original = inspect.getsource(native.fn)
    folded = _replace(original, {
        'def _fwd_grouped_kernel_stage1(': 'def _canonical_folded_stage1(',
        '    B_Seqlen,\n': '    B_Seqlen,\n    Query_Start,\n',
        '''    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)''':
        f'''    cur_batch = tl.program_id(0) // {query_tiles}
    query_tile = tl.program_id(0) % {query_tiles}
    cur_kv_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    query_first = tl.load(Query_Start + cur_batch)
    query_end = tl.load(Query_Start + cur_batch + 1)
    query_count = query_end - query_first
    tile_first = query_tile * 4
    if query_count <= tile_first:
        return
    offs_m = tl.arange(0, BLOCK_H)
    cur_query = query_first + tile_first + offs_m // 8
    cur_head = cur_kv_head * 8 + offs_m % 8
    mask_h = tile_first + offs_m // 8 < query_count''',
        '    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)\n':
        '''    request_seq_len = tl.load(B_Seqlen + cur_batch)
    first_causal = request_seq_len - query_count + tile_first + 1
    cur_batch_seq_len = request_seq_len - query_count + tl.minimum(tile_first + 4, query_count)
    if tl.cdiv(first_causal, NUM_KV_SPLITS) != tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS):
        return
    causal_len = request_seq_len - query_count + tile_first + offs_m // 8 + 1
''',
        'cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]':
        'cur_query[:, None] * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]',
        'mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")':
        'mask_h[:, None] & (offs_n[None, :] < split_kv_end) & (offs_n[None, :] < causal_len[:, None]), qk, float("-inf")',
        'cur_batch * stride_mid_ob\n            + cur_head[:, None] * stride_mid_oh':
        'cur_query[:, None] * stride_mid_ob\n            + cur_head[:, None] * stride_mid_oh',
        'cur_batch * stride_mid_ob\n            + cur_head * stride_mid_oh':
        'cur_query * stride_mid_ob\n            + cur_head * stride_mid_oh',
        '            acc / e_sum[:, None],':
        '            tl.where(causal_len[:, None] > split_kv_start, acc / e_sum[:, None], 0.),',
        '            e_max + tl.log(e_sum),':
        '            tl.where(causal_len > split_kv_start, e_max + tl.log(e_sum), -float("inf")),',
    })
    fallback = _replace(original, {
        'def _fwd_grouped_kernel_stage1(': 'def _canonical_boundary_stage1(',
        '    B_Seqlen,\n': '    B_Seqlen,\n    Request_Seq,\n    Query_Start,\n',
        '    IS_MLA: tl.constexpr = False,':
        '    REQUESTS: tl.constexpr,\n    REQUEST_BLOCK: tl.constexpr,\n    IS_MLA: tl.constexpr = False,',
        '    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)\n':
        '''    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    requests = tl.arange(0, REQUEST_BLOCK)
    starts = tl.load(Query_Start + requests, requests < REQUESTS, other=0)
    ends = tl.load(Query_Start + requests + 1, requests < REQUESTS, other=0)
    owns = (requests < REQUESTS) & (starts <= cur_batch) & (cur_batch < ends)
    owner = tl.max(tl.where(owns, requests + 1, 0), 0) - 1
    if owner < 0:
        return
    first = tl.sum(tl.where(owns, starts, 0), 0)
    count = tl.sum(tl.where(owns, ends - starts, 0), 0)
    request_seq = tl.load(Request_Seq + owner)
    tile_first = ((cur_batch - first) // 4) * 4
    first_causal = request_seq - count + tile_first + 1
    last_causal = request_seq - count + tl.minimum(tile_first + 4, count)
    if tl.cdiv(first_causal, NUM_KV_SPLITS) == tl.cdiv(last_causal, NUM_KV_SPLITS):
        return
''',
    })
    namespace = dict(native.fn.__globals__)
    _compile(folded, namespace, f'.folded_{query_tiles}.generated')
    _compile(fallback, namespace, '.boundary.generated')
    return namespace['_canonical_folded_stage1'], namespace['_canonical_boundary_stage1'], reduce


def forward(query, key_cache, value_cache, output, block_table, query_start_loc,
            seq_lens, sm_scale, k_scale, v_scale, *, max_query_len=8):
    if max_query_len == 1:
        return canonical.forward(query, key_cache, value_cache, output, block_table,
            query_start_loc, seq_lens, sm_scale, k_scale, v_scale, max_query_len=1)
    page_size = key_cache.shape[3] if key_cache.ndim == 5 else 0
    if (query.ndim != 3 or query.shape[1:] != (16, 256)
            or page_size not in (1120, 2240) or key_cache.shape[1:] != (2, 32, page_size, 8)
            or value_cache.ndim != 4 or value_cache.shape[1:] != (2, 256, page_size)
            or key_cache.shape[0] != value_cache.shape[0]
            or key_cache.shape[0] == 0 or output.shape != query.shape):
        raise ValueError('Canonical folded attention requires target H16/KV2/D256/page1120 or2240')
    if (not 1 <= max_query_len <= 16 or query.shape[0] > 128
            or not 1 <= seq_lens.numel() <= 8
            or block_table.ndim != 2 or block_table.shape[0] < seq_lens.numel()
            or block_table.shape[1] < 1 or query_start_loc.numel() != seq_lens.numel() + 1
            or not seq_lens.is_contiguous() or not query_start_loc.is_contiguous()
            or block_table.stride(1) != 1 or query.stride(2) != 1 or output.stride(2) != 1):
        raise ValueError('Unsupported query metadata or noncontiguous inner dimensions')
    if (not query.is_cuda or any(t.device != query.device for t in
            (key_cache, value_cache, output, block_table, query_start_loc, seq_lens, k_scale, v_scale))
            or any(t.dtype != torch.bfloat16 for t in (query, key_cache, value_cache, output))
            or any(t.dtype not in (torch.int32, torch.int64) for t in
                   (block_table, query_start_loc, seq_lens))):
        raise ValueError('Canonical folded attention requires BF16 and integer metadata on one GPU')
    rows = query.shape[0]
    if not rows:
        return output
    query_tiles = triton.cdiv(max_query_len, 4)
    folded, fallback, reduce = _kernels(query_tiles)
    table, seq = canonical._metadata(block_table, query_start_loc, seq_lens, rows, page_size)
    splits = 32
    logits = torch.empty((rows, 16, splits, 257), dtype=torch.float32, device=query.device)
    lse = torch.empty((rows, 16), dtype=torch.float32, device=query.device)
    options = dict(kv_group_num=8, q_head_num=16, BLOCK_DMODEL=256, BLOCK_DPE=0,
        BLOCK_DV=256, BLOCK_N=16, NUM_KV_SPLITS=splits,
        PAGE_SIZE=page_size, logit_cap=0., Lk=256, Lv=256, IS_MLA=False,
        stride_buf_kds=key_cache.stride(2), stride_buf_kxs=key_cache.stride(4),
        stride_buf_vds=value_cache.stride(2), num_warps=4, num_stages=1,
        waves_per_eu=1, matrix_instr_nonkdim=16, kpack=2)
    strides = (query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(3), key_cache.stride(1),
        value_cache.stride(0), value_cache.stride(3), value_cache.stride(1),
        logits.stride(0), logits.stride(1), logits.stride(2), k_scale, v_scale)
    folded[(seq_lens.numel() * query_tiles, 2, splits)](
        query, key_cache, value_cache, sm_scale, block_table, seq_lens, query_start_loc, logits,
        block_table.stride(0), *strides, BLOCK_H=32, **options)
    fallback[(rows, 2, splits)](
        query, key_cache, value_cache, sm_scale, table, seq, seq_lens, query_start_loc, logits,
        table.stride(0), *strides, BLOCK_H=16, REQUESTS=seq_lens.numel(),
        REQUEST_BLOCK=triton.next_power_of_2(seq_lens.numel()), **options)
    reduce(logits, query, output, lse, value_cache.transpose(2, 3), seq, splits)
    return output
