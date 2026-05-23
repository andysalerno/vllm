# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    BatchedTritonExperts,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8StaticTensorSym,
)


def test_batched_triton_experts_uses_platform_fp8_support():
    """Covers ROCm gfx12x through RocmPlatform.supports_fp8()."""
    with (
        patch(
            "vllm.model_executor.layers.fused_moe.experts."
            "fused_batched_moe.current_platform.is_cuda_alike",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.experts."
            "fused_batched_moe.current_platform.supports_fp8",
            return_value=True,
        ),
    ):
        assert BatchedTritonExperts._supports_quant_scheme(
            kFp8StaticTensorSym,
            kFp8DynamicTensorSym,
        )
