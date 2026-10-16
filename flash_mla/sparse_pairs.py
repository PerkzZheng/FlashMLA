from dataclasses import dataclass
from typing import Optional

import torch

_backend = torch.ops.flash_mla


@dataclass
class SparsePairMetadata:
    """Reusable device workspace for two adjacent, independently masked queries.

    Each input row is a set: valid indices must be unique within that row.
    Negative and out-of-bound indices are ignored. Pairing never crosses a
    batch boundary. Only even query lengths and top-k up to 4096 are supported.
    ``packed`` stores (physical_index << 2) | query_membership, padded with -1.
    Update this workspace whenever selections or lengths change, including
    inside a CUDA graph. Never share it between concurrently executing streams.

    ``sort`` accepts unsorted physical indices. ``merge`` sorts each row before
    merge-path; that preparation is part of update(), not a free prerequisite.
    ``bitmap`` requires each request's valid physical indices to lie in
    [request * request_stride, (request + 1) * request_stride), with a stride
    at most 65536. With stride zero, all requests share the whole KV domain,
    which must then contain at most 65536 tokens.
    """

    packed: torch.Tensor
    lengths: torch.Tensor
    scratch: torch.Tensor
    input_shape: tuple
    kv_bound: int
    request_stride: int
    algorithm: str

    def update(self, indices: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        if tuple(indices.shape) != self.input_shape:
            raise ValueError("Pair workspace cannot be reused with a different input shape")
        _backend.sparse_pair_union_out(
            indices, lengths, self.packed, self.lengths, self.scratch,
            self.kv_bound, self.request_stride,
            {"sort": 0, "merge": 1, "bitmap": 2}[self.algorithm],
        )
        return self


def prepare_sparse_pairs(
    indices: torch.Tensor,
    kv_bound: int,
    algorithm: str = "sort",
    request_stride: int = 0,
    lengths: Optional[torch.Tensor] = None,
) -> SparsePairMetadata:
    """Build graph-capturable paired metadata from contiguous int32 [B, SQ, K].

    Optional lengths are int32 [B, SQ], interpreted as valid input prefixes.
    Reuse the returned workspace with update() to avoid subsequent allocations.
    """
    if algorithm not in ("sort", "merge", "bitmap"):
        raise ValueError("algorithm must be sort, merge, or bitmap")
    if indices.ndim != 3 or not indices.is_cuda or indices.dtype != torch.int32:
        raise ValueError("indices must be a CUDA int32 [B, SQ, K] tensor")
    batch, queries, topk = indices.shape
    if not indices.is_contiguous() or batch < 1 or queries < 1 or queries % 2:
        raise ValueError("Pairing requires contiguous indices and nonempty even query lengths")
    if not 0 < topk <= 4096:
        raise ValueError("Pairing supports top-k 1..4096")
    capacity = (2 * topk + 63) // 64 * 64
    sorted_capacity = max(256, 1 << (topk - 1).bit_length())
    metadata = SparsePairMetadata(
        torch.empty((batch * queries // 2, 1, capacity), device=indices.device, dtype=torch.int32),
        torch.empty((batch * queries // 2,), device=indices.device, dtype=torch.int32),
        torch.empty((batch, queries, sorted_capacity) if algorithm == "merge" else (0,),
                    device=indices.device, dtype=torch.int32),
        tuple(indices.shape), kv_bound, request_stride, algorithm,
    )
    return metadata.update(indices, lengths)


def _check_pair_query(q, metadata):
    batch, queries, _ = metadata.input_shape
    if tuple(q.shape) != (batch, queries, 64, 512):
        raise ValueError("Paired attention requires [B, SQ, 64, 512] matching its metadata")
    if not q.is_contiguous() or q.dtype != torch.bfloat16 or q.device != metadata.packed.device:
        raise ValueError("Paired Q must be contiguous BF16 on the metadata device")
    return batch, queries


def flash_mla_with_kvcache_paired(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    metadata: SparsePairMetadata,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    force_no_split: bool = False,
    split_policy: str = "auto",
):
    """Opt-in SM100 M128 two-CTA sparse decode using prebuilt pair metadata.

    Q is [B, SQ, 64, 512]; KV uses the existing V4.1 FP8 paged layout.
    The union must already be up-to-date. Auto scheduling avoids splits for small
    unions or enough independent two-CTA clusters to fill the GPU. Otherwise a
    parallel planner rebuilds the schedule, including inside graph replay.
    split_policy="legacy" restores the original capacity-based split heuristic
    and serial planner for controlled comparisons. force_no_split overrides both.
    Returns output [B, SQ, 64, 512] and LSE [B, 64, SQ].
    """
    batch, queries = _check_pair_query(q, metadata)
    if split_policy not in ("auto", "legacy"):
        raise ValueError("split_policy must be auto or legacy")
    if k_cache.device != q.device or k_cache.ndim != 4:
        raise ValueError("KV cache must be a four-dimensional tensor on the Q device")
    if metadata.kv_bound > k_cache.shape[0] * k_cache.shape[1]:
        raise ValueError("Pair metadata KV bound exceeds the cache")
    if split_policy == "legacy" and not force_no_split and metadata.packed.shape[-1] > 640 and batch * queries // 2 > 4096:
        raise ValueError("Use force_no_split for more than 4096 query pairs")
    if attn_sink is not None and tuple(attn_sink.shape) != (64,):
        raise ValueError("attn_sink must have 64 elements")
    sink = torch.cat((attn_sink, attn_sink)) if attn_sink is not None else None
    output, lse, _, _ = _backend.sparse_decode_fwd(
        q.view(batch * queries // 2, 1, 128, 512), k_cache, metadata.packed,
        metadata.lengths, sink, None, None, None, None, None,
        512, softmax_scale if softmax_scale is not None else 512 ** -0.5, force_no_split, True,
        split_policy == "auto",
    )
    return output.view(batch, queries, 64, 512), lse.view(batch, queries, 64).transpose(1, 2)


def flash_mla_sparse_fwd_paired(
    q: torch.Tensor,
    kv: torch.Tensor,
    metadata: SparsePairMetadata,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
):
    """BF16 prefill variant with batched Q and a shared physical [KV, 1, 512] cache.

    Returns output [B, SQ, 64, 512], max logits [B, SQ, 64], LSE [B, SQ, 64].
    Membership and union lengths come from prepare_sparse_pairs, as for decode.
    """
    batch, queries = _check_pair_query(q, metadata)
    if kv.device != q.device or kv.ndim != 3 or metadata.kv_bound > kv.shape[0]:
        raise ValueError("KV must be a three-dimensional cache covering the metadata bound")
    if attn_sink is not None and tuple(attn_sink.shape) != (64,):
        raise ValueError("attn_sink must have 64 elements")
    sink = torch.cat((attn_sink, attn_sink)) if attn_sink is not None else None
    output, maximum, lse = _backend.sparse_prefill_fwd(
        q.view(-1, 128, 512), kv, metadata.packed,
        softmax_scale if softmax_scale is not None else 512 ** -0.5,
        512, sink, metadata.lengths, True,
    )
    return output.view(batch, queries, 64, 512), maximum.view(batch, queries, 64), lse.view(batch, queries, 64)
