# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Unit tests for the HiCacheFile multi-page span contract.

Compressed-DSA pools inflate the radix-tree page so one compressed index row
stays atomic; in span mode one storage key covers K consecutive host pages.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
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


def _controller_storage_config(storage_page_size: int) -> HiCacheStorageConfig:
    """Run the controller's real config-generation path.

    The shell is built the same way test_hybrid_dsa_hicache.py builds
    controllers (object.__new__ + parallel getters patched); only the
    parallel-state globals are stubbed, every config field is produced by
    HiCacheController._generate_storage_config itself.
    """
    from sglang.srt.managers import cache_controller as cc_module

    controller = object.__new__(cc_module.HiCacheController)
    controller.storage_page_size = storage_page_size
    controller.enable_storage_metrics = False
    controller.mem_pool_host = SimpleNamespace(layout="layer_first")
    controller.mem_pool_device = object.__new__(MLATokenToKVPool)
    controller.get_attn_cp_rank_and_size = lambda: (0, 1)

    parallel = SimpleNamespace(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_tp_rank=0, attn_tp_size=1
    )
    with (
        patch.object(cc_module, "is_dp_attention_enabled", return_value=False),
        patch.object(cc_module, "get_parallel", lambda: parallel),
        patch.object(cc_module, "get_attention_dp_rank", lambda: 0),
    ):
        config = controller._generate_storage_config(model_name="span-ctl-test")
    assert config.storage_page_size == storage_page_size
    return config


@pytest.mark.parametrize(
    "storage_page_size,pages_per_key",
    [(4, 1), (16, 4)],  # degenerate non-span page, inflated span page
)
def test_mixed_kv_mamba_roundtrip_controller_config(
    tmp_path, storage_page_size, pages_per_key
):
    """Mixed KV + Mamba roundtrip under the controller-generated config, both
    for the degenerate (storage_page_size == page_size) and the span
    (storage_page_size == 4 * page_size) wiring.

    Regression for the review finding: a uniform stride expected Mamba to
    carry storage_page_size slots per key, rejecting its one-checkpoint-slot
    transfers in span mode *and* in ordinary non-span hybrid file storage.
    """
    config = _controller_storage_config(storage_page_size)
    backend = HiCacheFile(
        config, file_path=str(tmp_path / f"hicache-{storage_page_size}")
    )
    kv_pool = FakeSpanHostPool(num_pages=8)
    state_pool = FakeStatePool(num_slots=8)
    backend.register_mem_host_pool_v2(kv_pool, PoolName.KV)
    backend.register_mem_host_pool_v2(state_pool, PoolName.MAMBA)

    keys = ["h0", "h1"]
    kv_indices = torch.arange(2 * storage_page_size)
    state_indices = torch.tensor([3, 5])
    transfers = [
        PoolTransfer(name=PoolName.KV, host_indices=kv_indices, keys=keys),
        PoolTransfer(name=PoolName.MAMBA, host_indices=state_indices, keys=keys),
    ]

    for k in range(2):
        for j in range(pages_per_key):
            slot = k * storage_page_size + j * PAGE_SIZE
            kv_pool.kv_buffer[:, slot : slot + PAGE_SIZE, :, :] = 10.0 * (k + 1) + j
    state_pool.states[3] = 111.0
    state_pool.states[5] = 222.0

    res = backend.batch_set_v2(transfers)
    assert all(res[PoolName.KV]) and all(res[PoolName.MAMBA])

    kv_pool.kv_buffer.zero_()
    state_pool.states.zero_()
    res = backend.batch_get_v2(transfers)
    assert all(res[PoolName.KV]) and all(res[PoolName.MAMBA])
    for k in range(2):
        for j in range(pages_per_key):
            slot = k * storage_page_size + j * PAGE_SIZE
            assert torch.all(
                kv_pool.kv_buffer[:, slot : slot + PAGE_SIZE, :, :]
                == 10.0 * (k + 1) + j
            )
    assert torch.all(state_pool.states[3] == 111.0)
    assert torch.all(state_pool.states[5] == 222.0)


@pytest.mark.parametrize("all_sidecars_ok", [True, False])
def test_backup_skip_completed_tokens_uses_storage_page(all_sidecars_ok):
    """backup_skip accounting must count one storage page per hash key.

    With span mode a single key covers storage_page_size tokens (e.g. 256),
    not page_size (64); the sidecar-ok and sidecar-failed branches must both
    report against the storage page so backup-token metrics agree across
    ranks.
    """
    controller = object.__new__(HybridCacheController)
    controller.backup_skip = True
    controller.page_size = PAGE_SIZE
    controller.storage_page_size = STORAGE_PAGE_SIZE
    controller.storage_backend_type = "file"
    controller.mem_pool_host = MagicMock()
    controller.storage_backend = MagicMock()
    results = {PoolName.MAMBA: [True, True] if all_sidecars_ok else [True, False]}
    controller.storage_backend.batch_set_v2 = MagicMock(return_value=results)

    operation = SimpleNamespace(
        pool_transfers=[
            PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=torch.tensor([3, 5]),
                keys=["h0", "h1"],
            )
        ],
        hash_value=["h0", "h1"],
        completed_tokens=0,
        pool_storage_result=MagicMock(),
        host_indices=None,
    )
    controller._page_backup(operation)

    expected = len(operation.hash_value) * STORAGE_PAGE_SIZE if all_sidecars_ok else 0
    assert operation.completed_tokens == expected
