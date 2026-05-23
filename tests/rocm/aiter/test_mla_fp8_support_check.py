# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for AITER MLA FP8 support detection.

These tests verify that the _check_aiter_mla_fp8_support() function
correctly handles various error conditions without crashing.
"""

import sys
from types import ModuleType
from unittest.mock import patch

import pytest


def _fake_rocm_module(*, on_mi3xx: bool, on_gfx12x: bool) -> ModuleType:
    module = ModuleType("vllm.platforms.rocm")
    module.on_mi3xx = lambda: on_mi3xx
    module.on_gfx12x = lambda: on_gfx12x
    return module


def test_aiter_supports_gfx12x():
    """Test that AITER is considered available on gfx12x/RDNA4."""
    from vllm._aiter_ops import is_aiter_found_and_supported

    with (
        patch("vllm._aiter_ops.IS_AITER_FOUND", True),
        patch("vllm._aiter_ops.current_platform.is_rocm", return_value=True),
        patch.dict(
            sys.modules,
            {
                "vllm.platforms.rocm": _fake_rocm_module(
                    on_mi3xx=False,
                    on_gfx12x=True,
                )
            },
        ),
    ):
        assert is_aiter_found_and_supported() is True


def test_aiter_rejects_unsupported_rocm_arch():
    """Test that AITER remains disabled on unsupported ROCm architectures."""
    from vllm._aiter_ops import is_aiter_found_and_supported

    with (
        patch("vllm._aiter_ops.IS_AITER_FOUND", True),
        patch("vllm._aiter_ops.current_platform.is_rocm", return_value=True),
        patch.dict(
            sys.modules,
            {
                "vllm.platforms.rocm": _fake_rocm_module(
                    on_mi3xx=False,
                    on_gfx12x=False,
                )
            },
        ),
    ):
        assert is_aiter_found_and_supported() is False


class TestAiterMlaFp8SupportCheck:
    """Test cases for _check_aiter_mla_fp8_support() function."""

    def setup_method(self):
        """Reset the global cache before each test."""
        import vllm._aiter_ops as aiter_ops

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

    @patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True)
    def test_import_error_handling(self, mock_supported):
        """Test that ImportError is handled gracefully."""
        import vllm._aiter_ops as aiter_ops
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

        # Should return False without raising
        with patch(
            "inspect.signature",
            side_effect=ImportError("No module"),
        ):
            result = _check_aiter_mla_fp8_support()
            assert result is False

    @patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True)
    def test_module_not_found_error_handling(self, mock_supported):
        """Test that ModuleNotFoundError is handled gracefully."""
        import vllm._aiter_ops as aiter_ops
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

        with patch(
            "inspect.signature",
            side_effect=ModuleNotFoundError("Module not found"),
        ):
            # Should return False without raising
            assert _check_aiter_mla_fp8_support() is False
            # Cache should be set to False
            assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is False

    @patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True)
    def test_attribute_error_handling(self, mock_supported):
        """Test that AttributeError is handled gracefully."""
        import vllm._aiter_ops as aiter_ops
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

        with patch(
            "inspect.signature",
            side_effect=AttributeError("No attribute"),
        ):
            assert _check_aiter_mla_fp8_support() is False
            assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is False

    @patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True)
    def test_value_error_handling(self, mock_supported):
        """Test that ValueError is handled gracefully (no signature)."""
        import vllm._aiter_ops as aiter_ops
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

        with patch(
            "inspect.signature",
            side_effect=ValueError("No signature"),
        ):
            assert _check_aiter_mla_fp8_support() is False
            assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is False

    @patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True)
    def test_type_error_handling(self, mock_supported):
        """Test that TypeError is handled gracefully (not callable)."""
        import vllm._aiter_ops as aiter_ops
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

        with patch(
            "inspect.signature",
            side_effect=TypeError("Not a callable"),
        ):
            assert _check_aiter_mla_fp8_support() is False
            assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is False

    @patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True)
    def test_result_caching(self, mock_supported):
        """Test that the result is cached after first check."""
        import vllm._aiter_ops as aiter_ops

        # Set cache to True
        aiter_ops._AITER_MLA_SUPPORTS_FP8 = True

        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        # Should return cached value without re-checking
        result = _check_aiter_mla_fp8_support()
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
