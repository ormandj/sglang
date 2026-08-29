import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPagedAllocatorSnapshot(unittest.TestCase):
    @staticmethod
    def _allocator() -> PagedTokenToKVPoolAllocator:
        allocator = PagedTokenToKVPoolAllocator.__new__(PagedTokenToKVPoolAllocator)
        allocator.num_pages = 8
        allocator.free_pages = torch.arange(1, 9, dtype=torch.int64)
        allocator._debug_free_pages_snapshot = None
        allocator._debug_free_pages_metadata = None
        allocator._debug_free_pages_boundary = None
        return allocator

    @patch("sglang.srt.envs.SGLANG_ENABLE_ASYNC_ASSERT.get", return_value=True)
    def test_unchanged_state_passes(self, _):
        allocator = self._allocator()
        allocator._debug_capture_free_pages("after first operation")
        allocator._debug_check_free_pages("before second operation")

    @patch("sglang.srt.envs.SGLANG_ENABLE_ASYNC_ASSERT.get", return_value=True)
    def test_change_between_operations_is_attributed(self, _):
        allocator = self._allocator()
        allocator._debug_capture_free_pages("after first operation")
        allocator.free_pages[3] = 7

        with self.assertRaisesRegex(
            AssertionError,
            "previous=after first operation current=before second operation",
        ):
            allocator._debug_check_free_pages("before second operation")


if __name__ == "__main__":
    unittest.main()
