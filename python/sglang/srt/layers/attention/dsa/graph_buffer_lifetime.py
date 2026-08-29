"""Bounded diagnostics for DSA buffers recorded by full CUDA graphs.

CUDA graphs retain device addresses, not Python tensor owners.  The probe mode
records addresses without changing lifetimes; retain mode additionally keeps
the captured tensors alive so the two behaviours can be compared using one
immutable image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

_ENV = "SGLANG_DSA_GRAPH_BUFFER_LIFETIME"
_VALID_MODES = frozenset(("off", "probe", "retain"))
_MAX_RECORDS = 4096


@dataclass(frozen=True)
class GraphBufferRecord:
    label: str
    device: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    data_start: int
    data_end: int
    storage_start: int
    storage_end: int


_records: list[GraphBufferRecord] = []
_owners: list[torch.Tensor] = []


def _mode() -> str:
    mode = os.environ.get(_ENV, "off").lower()
    if mode not in _VALID_MODES:
        raise RuntimeError(
            f"{_ENV} must be one of {sorted(_VALID_MODES)}, got {mode!r}"
        )
    return mode


def _logical_end(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return tensor.data_ptr()
    max_offset = sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    return tensor.data_ptr() + (max_offset + 1) * tensor.element_size()


def _record(tensor: torch.Tensor, label: str) -> GraphBufferRecord:
    storage = tensor.untyped_storage()
    storage_start = storage.data_ptr()
    return GraphBufferRecord(
        label=label,
        device=str(tensor.device),
        dtype=str(tensor.dtype),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        data_start=tensor.data_ptr(),
        data_end=_logical_end(tensor),
        storage_start=storage_start,
        storage_end=storage_start + storage.nbytes(),
    )


def register_graph_buffer(tensor: torch.Tensor, label: str) -> None:
    """Record a DSA intermediate only during the real CUDA graph capture."""
    mode = _mode()
    if mode == "off" or not tensor.is_cuda:
        return
    if not torch.cuda.is_current_stream_capturing():
        return
    if len(_records) >= _MAX_RECORDS:
        raise RuntimeError(f"{_ENV} exceeded {_MAX_RECORDS} captured buffers")
    _records.append(_record(tensor, label))
    if mode == "retain":
        _owners.append(tensor)


def _range_gap(start: int, end: int, other_start: int, other_end: int) -> int:
    if max(start, other_start) < min(end, other_end):
        return -1
    if end <= other_start:
        return other_start - end
    return start - other_end


def describe_graph_buffer_overlap(tensor: torch.Tensor) -> str:
    """Describe exact overlaps and nearest captured DSA buffer ranges."""
    if not _records:
        return "none-recorded"
    target = _record(tensor, "target")
    ranked: list[tuple[int, int, GraphBufferRecord]] = []
    for record in _records:
        logical_gap = _range_gap(
            target.data_start, target.data_end, record.data_start, record.data_end
        )
        storage_gap = _range_gap(
            target.data_start,
            target.data_end,
            record.storage_start,
            record.storage_end,
        )
        relation = 0 if logical_gap < 0 else 1 if storage_gap < 0 else 2
        gap = logical_gap if relation == 0 else storage_gap
        ranked.append((relation, gap, record))
    ranked.sort(key=lambda item: (item[0], item[1]))
    parts = []
    for relation, gap, record in ranked[:8]:
        relation_name = ("logical-overlap", "storage-overlap", "nearest")[relation]
        parts.append(
            f"{relation_name}:{record.label}:gap={gap}:"
            f"data=[{record.data_start},{record.data_end}):"
            f"storage=[{record.storage_start},{record.storage_end}):"
            f"dtype={record.dtype}:shape={record.shape}:stride={record.stride}"
        )
    return " | ".join(parts)


def _reset_for_test() -> None:
    _records.clear()
    _owners.clear()


def _state_for_test() -> tuple[list[GraphBufferRecord], list[Any]]:
    return list(_records), list(_owners)
