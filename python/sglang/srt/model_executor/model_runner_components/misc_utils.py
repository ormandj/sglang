from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def resolve_pp_proxy_topk_size(
    *, model_config: ModelConfig, pp_size: int, pp_rank: int, start_layer: int
) -> Optional[int]:
    hf_config = model_config.hf_text_config
    if (
        pp_size <= 1
        or pp_rank == 0
        or not is_deepseek_dsa(hf_config)
        or not dsa_layer_skips_topk(hf_config, start_layer)
    ):
        return None
    return getattr(hf_config, "index_topk", None)


def create_msprobe_debugger(server_args: ServerArgs) -> Optional[Any]:
    if server_args.msprobe_dump_config is None:
        return None

    try:
        from msprobe.pytorch import PrecisionDebugger, seed_all
    except ImportError:
        logger.warning(
            "Please install msprobe for tensor data dump: pip install mindstudio-probe --pre, "
            "see https://gitcode.com/Ascend/msprobe for details."
        )
        return None

    seed_all(mode=True)
    return PrecisionDebugger(config_path=server_args.msprobe_dump_config)
