# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Unit tests for the HiCacheFile multi-page span contract.

Compressed-DSA pools inflate the radix-tree page so one compressed index row
stays atomic; in span mode one storage key covers K consecutive host pages.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.utils import get_hash_str

PAGE_SIZE = 4
STORAGE_PAGE_SIZE = 16  # 4 host pages per key
PAGES_PER_KEY = STORAGE_PAGE_SIZE // PAGE_SIZE
DIM = 3
LAYER_NUM = 2
PAGE_NUMEL = LAYER_NUM * PAGE_SIZE * DIM
STATE_NUMEL = 7


class FakeSpanHostPool:
    """Minimal page-aligned host pool mimic (layer_first semantics)."""

    page_size = PAGE_SIZE

    def __init__(self, num_pages: int):
        self.kv_buffer = torch.zeros(LAYER_NUM, num_pages * PAGE_SIZE, 1, DIM)

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        page = self.kv_buffer[:, index : index + PAGE_SIZE, :, :]
        return page.flatten() if flat else page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(PAGE_NUMEL)

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        self.kv_buffer[:, index : index + PAGE_SIZE, :, :] = data_page.reshape(
            LAYER_NUM, PAGE_SIZE, 1, DIM
        )


class FakeStatePool:
    """Mamba-like state pool: one fixed-size state slot per key."""

    page_size = 1

    def __init__(self, num_slots: int):
        self.states = torch.zeros(num_slots, STATE_NUMEL)

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        return self.states[index].clone()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(STATE_NUMEL)

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        self.states[index] = data_page


@pytest.fixture
def span_setup(tmp_path):
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="span-test",
        storage_page_size=STORAGE_PAGE_SIZE,
    )
    backend = HiCacheFile(config, file_path=str(tmp_path / "hicache"))
    pool = FakeSpanHostPool(num_pages=8)
    backend.register_mem_host_pool_v2(pool, PoolName.KV)
    return backend, pool


def _key_slots(key_idx: int) -> torch.Tensor:
    start = key_idx * STORAGE_PAGE_SIZE
    return torch.arange(start, start + STORAGE_PAGE_SIZE)


def _fill_pages(pool, key_idx: int, value_base: float) -> None:
    for j in range(PAGES_PER_KEY):
        slot = key_idx * STORAGE_PAGE_SIZE + j * PAGE_SIZE
        pool.kv_buffer[:, slot : slot + PAGE_SIZE, :, :] = value_base + j


def _assert_pages(pool, key_idx: int, value_base: float) -> None:
    for j in range(PAGES_PER_KEY):
        slot = key_idx * STORAGE_PAGE_SIZE + j * PAGE_SIZE
        assert torch.all(
            pool.kv_buffer[:, slot : slot + PAGE_SIZE, :, :] == value_base + j
        )


def test_span_roundtrip(span_setup):
    backend, pool = span_setup
    keys = ["h0", "h1"]
    host_indices = torch.cat([_key_slots(0), _key_slots(1)])
    transfers = [PoolTransfer(name=PoolName.KV, host_indices=host_indices, keys=keys)]

    _fill_pages(pool, 0, 10.0)
    _fill_pages(pool, 1, 20.0)
    assert all(backend.batch_set_v2(transfers)[PoolName.KV])

    pool.kv_buffer.zero_()
    assert all(backend.batch_get_v2(transfers)[PoolName.KV])
    _assert_pages(pool, 0, 10.0)
    _assert_pages(pool, 1, 20.0)


def test_span_size_mismatch(span_setup):
    backend, pool = span_setup
    transfers = [
        PoolTransfer(name=PoolName.KV, host_indices=torch.arange(4), keys=["h0"])
    ]
    assert backend.batch_set_v2(transfers)[PoolName.KV] == [False]


def test_batch_exists_v2_prefix(span_setup):
    backend, pool = span_setup
    keys = ["h0", "h1"]
    host_indices = torch.cat([_key_slots(0), _key_slots(1)])
    transfers = [PoolTransfer(name=PoolName.KV, host_indices=host_indices, keys=keys)]
    assert backend.batch_set_v2(transfers)[PoolName.KV] == [True, True]
    assert backend.batch_exists_v2(keys, transfers).kv_hit_pages == 2
    assert backend.batch_exists_v2(["h0", "missing"], transfers).kv_hit_pages == 1


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="HiCache native hash requires little-endian Linux",
)
def test_hash_chain_alignment():
    tokens = list(range(512))
    hashes = get_hash_str(tokens, None, page_size=256)
    assert isinstance(hashes, list) and len(hashes) == 2
    tail = get_hash_str(tokens[256:], hashes[0], page_size=256)
    assert tail == [hashes[1]]


def test_mixed_kv_and_state_pool_roundtrip(span_setup):
    """KV carries span slots per key; state pools carry one entry per key.

    Regression: _batch_io_v2 used to demand the KV span stride from every
    pool, so mamba-style one-slot-per-key transfers were rejected and their
    storage objects never written.
    """
    backend, kv_pool = span_setup
    state_pool = FakeStatePool(num_slots=8)
    backend.register_mem_host_pool_v2(state_pool, PoolName.MAMBA)
    keys = ["h0", "h1"]
    kv_indices = torch.cat([_key_slots(0), _key_slots(1)])
    state_indices = torch.tensor([3, 5])
    transfers = [
        PoolTransfer(name=PoolName.KV, host_indices=kv_indices, keys=keys),
        PoolTransfer(name=PoolName.MAMBA, host_indices=state_indices, keys=keys),
    ]

    _fill_pages(kv_pool, 0, 10.0)
    _fill_pages(kv_pool, 1, 20.0)
    state_pool.states[3] = 111.0
    state_pool.states[5] = 222.0
    res = backend.batch_set_v2(transfers)
    assert all(res[PoolName.KV]) and all(res[PoolName.MAMBA])

    kv_pool.kv_buffer.zero_()
    state_pool.states.zero_()
    res = backend.batch_get_v2(transfers)
    assert all(res[PoolName.KV]) and all(res[PoolName.MAMBA])
    _assert_pages(kv_pool, 0, 10.0)
    _assert_pages(kv_pool, 1, 20.0)
    assert torch.all(state_pool.states[3] == 111.0)
    assert torch.all(state_pool.states[5] == 222.0)


def test_backend_supports_page_spans():
    assert StorageBackendFactory.backend_supports_page_spans("file") is True
    assert StorageBackendFactory.backend_supports_page_spans("nonexistent") is False
