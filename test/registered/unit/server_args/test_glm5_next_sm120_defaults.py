import unittest
from unittest.mock import patch

from sglang.srt.arg_groups import model_hook
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlm5NextSm120Defaults(unittest.TestCase):
    def test_disables_deepgemm_hc_prenorm_for_glm5_next_on_sm120(self):
        with (
            patch.object(model_hook, "is_sm120_supported", return_value=True),
            patch.object(envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM, "set") as set_flag,
        ):
            model_hook._apply_glm5_next_sm120_defaults(
                "Glm5NextForConditionalGeneration"
            )

        set_flag.assert_called_once_with(False)

    def test_leaves_other_architectures_unchanged(self):
        with (
            patch.object(model_hook, "is_sm120_supported", return_value=True),
            patch.object(envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM, "set") as set_flag,
        ):
            model_hook._apply_glm5_next_sm120_defaults("DeepseekV3ForCausalLM")

        set_flag.assert_not_called()

    def test_leaves_glm5_next_unchanged_off_sm120(self):
        with (
            patch.object(model_hook, "is_sm120_supported", return_value=False),
            patch.object(envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM, "set") as set_flag,
        ):
            model_hook._apply_glm5_next_sm120_defaults(
                "Glm5NextForConditionalGeneration"
            )

        set_flag.assert_not_called()


if __name__ == "__main__":
    unittest.main()
