import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import (
    _get_compress_gate_stream,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSAKPoolSharedCompressionStream(unittest.TestCase):
    def test_cuda_indexers_use_one_named_stream(self):
        alt_stream = MagicMock()
        shared_stream = MagicMock()
        with patch(
            "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.is_cuda",
            return_value=True,
        ), patch(
            "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.get_stream",
            return_value=shared_stream,
        ) as get_stream:
            first = _get_compress_gate_stream(alt_stream)
            second = _get_compress_gate_stream(alt_stream)

        self.assertIs(first, shared_stream)
        self.assertIs(second, shared_stream)
        get_stream.assert_called_with("dsa_index_compress_gate")

    def test_no_stream_without_cuda_or_alt_stream(self):
        with patch(
            "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.is_cuda",
            return_value=False,
        ), patch(
            "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.get_stream"
        ) as get_stream:
            self.assertIsNone(_get_compress_gate_stream(MagicMock()))
            get_stream.assert_not_called()

        with patch(
            "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.is_cuda",
            return_value=True,
        ), patch(
            "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.get_stream"
        ) as get_stream:
            self.assertIsNone(_get_compress_gate_stream(None))
            get_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
