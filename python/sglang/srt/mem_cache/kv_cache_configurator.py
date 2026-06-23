from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner_components.pool_configurator import (
        MemoryPoolConfig,
    )


class KVCacheConfigResult(msgspec.Struct, frozen=True, kw_only=True):
    """Configurator output — caller writes back to ModelRunner fields."""

    max_total_num_tokens: int
    max_running_requests: int
    full_max_total_num_tokens: Optional[int]
    swa_max_total_num_tokens: Optional[int]
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: KVCache
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    memory_pool_config: MemoryPoolConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class KVCacheConfigurator:
    """KV cache pipeline (profile -> resolve -> constrain -> init pools).

    Replaces ``ModelRunnerKVCacheMixin`` via composition. ``frozen=True``
    blocks any stale ``self.X = Y`` writes left over from the mixin
    migration; ``slots=True`` blocks attribute typos at runtime;
    ``kw_only=True`` forces named-kwargs construction at the caller.

    Pipeline intermediate state (profiled bytes / resolved configs / pool
    objects) flows through local variables + return values, not via
    attribute writes on ``self``.
    """

    # deployment env (runtime, not in server_args)
    device: str
    gpu_id: int
    # model / dtype (resolved objects, not in server_args)
    model_config: ModelConfig
    server_args: ServerArgs
    kv_cache_dtype: torch.dtype
    # speculative decoding (runtime / derived, not in server_args)
    spec_algorithm: SpeculativeAlgorithm
    is_draft_worker: bool
    dflash_draft_num_layers: Optional[int]
    # arch flags (derived, not direct server_args fields)
    is_hybrid_swa: bool
    is_hybrid_swa_compress: bool
    use_mla_backend: bool
    mambaish_config: Optional[Any]
    hybrid_gdn_config: Optional[Any]
    # PP slice
    start_layer: int
    end_layer: int
    num_effective_layers: int
    # optional pre-injection (draft worker reuses target's pool)
    req_to_token_pool: Optional[ReqToTokenPool]
    token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator]
    # draft worker budget
    memory_pool_config: Optional[MemoryPoolConfig]

    def configure(self, *, pre_model_load_memory: int) -> KVCacheConfigResult:
        raise NotImplementedError("populated in kvc-migrate-method-bodies")
