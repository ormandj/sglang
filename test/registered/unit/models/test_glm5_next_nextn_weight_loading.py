import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_nextn import DeepseekModelNextN
from sglang.srt.models.glm5_next_nextn import (
    Glm5NextForConditionalGenerationNextN,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, loaded_weight):
        self.loaded = loaded_weight


class _FakeModelOptFp4Config:
    def get_name(self):
        return "modelopt_fp4"


class TestGlm5NextNextNWeightLoading(unittest.TestCase):
    def test_deepseek_modelopt_fp4_default_remains_unquantized(self):
        quant_config = _FakeModelOptFp4Config()

        resolved = DeepseekModelNextN._resolve_modelopt_fp4_quant_config(
            quant_config, preserve_modelopt_fp4_quant_config=False
        )

        self.assertIsNone(resolved)

    def test_glm_can_preserve_modelopt_fp4_in_nextn_constructor(self):
        quant_config = _FakeModelOptFp4Config()

        resolved = DeepseekModelNextN._resolve_modelopt_fp4_quant_config(
            quant_config,
            preserve_modelopt_fp4_quant_config=(
                Glm5NextForConditionalGenerationNextN.preserve_modelopt_fp4_nextn
            ),
        )

        self.assertIs(resolved, quant_config)

    def test_quantized_glm_nextn_preserves_modelopt_fp4(self):
        quant_config = _FakeModelOptFp4Config()
        config = SimpleNamespace(
            num_hidden_layers=45,
            quantization_config={"ignore": ["*.self_attn.*"]},
        )

        resolved = Glm5NextForConditionalGenerationNextN.__new__(
            Glm5NextForConditionalGenerationNextN
        )._resolve_nextn_quant_config(config, quant_config)

        self.assertIs(resolved, quant_config)
        self.assertTrue(
            Glm5NextForConditionalGenerationNextN.preserve_modelopt_fp4_nextn
        )

    def test_bf16_glm_nextn_still_drops_modelopt_fp4(self):
        quant_config = _FakeModelOptFp4Config()
        config = SimpleNamespace(
            num_hidden_layers=45,
            quantization_config={"ignore": ["model.layers.45.*"]},
        )

        resolved = Glm5NextForConditionalGenerationNextN.__new__(
            Glm5NextForConditionalGenerationNextN
        )._resolve_nextn_quant_config(config, quant_config)

        self.assertIsNone(resolved)

    def test_checkpoint_qkv_sources_load_fused_projection(self):
        fused_param = _FakeParam()
        model = SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=45,
                num_nextn_predict_layers=1,
                n_routed_experts=0,
                q_lora_rank=1536,
            ),
            model=SimpleNamespace(
                decoder=SimpleNamespace(self_attn=None),
            ),
            named_parameters=lambda: iter(
                [
                    (
                        "model.decoder.self_attn.fused_qkv_a_proj_with_mqa.weight",
                        fused_param,
                    )
                ]
            ),
            num_fused_shared_experts=0,
            quant_config=None,
        )
        q_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        kv_weight = torch.arange(3, dtype=torch.float32).reshape(1, 3) + 10

        Glm5NextForConditionalGenerationNextN.load_weights(
            model,
            [
                ("model.layers.45.self_attn.q_a_proj.weight", q_weight),
                (
                    "model.layers.45.self_attn.kv_a_proj_with_mqa.weight",
                    kv_weight,
                ),
            ],
        )

        self.assertTrue(model.fuse_qkv_a_proj)
        torch.testing.assert_close(
            fused_param.loaded,
            torch.cat([q_weight, kv_weight], dim=0),
        )


if __name__ == "__main__":
    unittest.main()
