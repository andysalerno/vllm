# GFX12x ROCm enablement notes

This file captures context that is useful when expanding AMD RDNA4 / gfx12x
support in vLLM, especially for AMD Radeon AI PRO R9700 (`gfx1201`). It is a
working handoff for future developers and agents, not a support matrix.

## Hardware and architecture facts

- AMD Radeon AI PRO R9700 reports as `gfx1201`.
- `gfx1201` is RDNA4 / gfx12x, not CDNA. Do not assume every MI300/CDNA path is
  safe on this GPU.
- RDNA4 has substantial overlap with gfx11xx and some older/CDNA-oriented ROCm
  functionality, but it is not fully backward compatible. Prefer capability
  checks or explicitly validated gfx12x gates over broad "CDNA or newer"
  assumptions.
- A local R9700 machine can confirm detection with:

  ```bash
  rocminfo | grep -E 'Name: *gfx|Marketing Name|Uuid'
  ```

## Existing vLLM gfx12x plumbing

The repository already has several important gfx12x pieces. Check these before
adding new architecture logic:

- `CMakeLists.txt` includes `gfx1200` and `gfx1201` in `HIP_SUPPORTED_ARCHS`.
- `docker/Dockerfile.rocm_base` includes `gfx1200` and `gfx1201` in
  `PYTORCH_ROCM_ARCH`.
- `vllm/platforms/rocm.py` maps device ID `0x7551` to
  `AMD_Radeon_R9700`.
- `vllm/platforms/rocm.py` exposes helpers such as `on_gfx12x()`,
  `on_gfx1x()`, `on_gfx9()`, and `on_mi3xx()`.
- `RocmPlatform.supports_fp8()` already returns true for `on_gfx9()` or
  `on_gfx12x()`. Prefer this platform method for general FP8 feature gates.

## Useful ROCm helper semantics

As of this note, the central ROCm helpers behave approximately like this:

- `on_gfx12x()` detects gfx12xx / RDNA4.
- `on_gfx1x()` detects gfx11xx and gfx12xx.
- `on_mi3xx()` detects MI300-family targets such as `gfx942` and `gfx950`.
- `on_gfx9()` includes `gfx90a`, `gfx942`, and `gfx950`.

When adding a gate, choose the narrowest helper that matches the validated
kernel behavior. `on_gfx1x()` is useful for RDNA3/RDNA4 code paths, while
`on_mi3xx()` remains the safer choice for kernels that are known only on
MI300-class devices.

## Areas recently enabled for gfx12x

The following areas were found to have stale gates that excluded R9700 even
though the surrounding code already had gfx12x-compatible paths:

- Global AITER availability in `vllm/_aiter_ops.py`: allow MI3xx or gfx12x.
- ROCm AITER Flash Attention capability checks in
  `vllm/v1/attention/backends/rocm_aiter_fa.py`: use ROCm GCN helpers instead
  of relying on CUDA-style device capability.
- ViT ROCm attention backend selection in `vllm/platforms/rocm.py`: allow AITER
  Flash Attention on MI3xx or gfx12x when AITER is enabled.
- Batched Triton MoE FP8 support in
  `vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py`: reuse
  `current_platform.supports_fp8()` instead of duplicating arch checks.

## Known code paths that already had gfx12x coverage

Before changing nearby gates, check whether a related path already handles
gfx12x:

- Standard Triton MoE experts use `current_platform.supports_fp8()`.
- GPT-OSS / OAI Triton MoE support already includes the gfx1x family
  (`gfx11xx` and `gfx12xx`).
- ROCm scaled-mm kernel support already allows MI3xx or gfx12x.
- RDNA ViT attention selection already recognizes RDNA3/RDNA4 Triton Flash
  Attention paths.
- Tuned R9700 FP8 MoE configs may already exist for common model shapes; avoid
  adding duplicate config entries without checking current configs.

## Cautions and non-goals

- Do not broadly replace MI3xx/CDNA checks with gfx12x. Some kernels use
  instructions, wave behavior, or tuning assumptions that are not portable to
  RDNA4.
- Do not assume gfx950/CDNA4-specific FP4 or MXFP4 paths are valid on gfx1201
  without hardware validation.
- Be careful with AITER: enabling the top-level package gate can expose more
  AITER-decorated features. Keep stricter per-kernel gates in place unless each
  kernel is validated.
- Avoid adding fallback paths from stale PRs without re-checking them against
  current `main`; some were workarounds for code that has since changed.
- Prefer `RocmPlatform.supports_fp8()` for broad FP8 feature selection. Use
  explicit arch checks only when the kernel has tighter requirements than
  platform FP8 support.

## Testing tips

- Use `rocminfo` to confirm the machine really sees R9700 devices as `gfx1201`.
- For lightweight Python validation, tests can patch ROCm helper functions
  instead of importing real ROCm modules in a CPU-only environment.
- ROCm-marked tests may skip in a CPU PyTorch environment even when selector
  logic is covered by mocks.
- When testing in this repository, follow `AGENTS.md`: use `uv` and
  `.venv/bin/python`; do not use system `python3` or bare `pip`.

## PR and history references to check

The following prior work is useful context, but may be stale or partially
superseded:

- `vllm-project/vllm#37826`: widened OAI/GPT-OSS Triton MoE capability to
  include gfx12 / RDNA4. Much of this is already present.
- `vllm-project/vllm#36659`: proposed R9700 / gfx1201 FP8 and AITER enablement.
  Useful for goals and affected areas, but do not copy blindly.
- `vllm-project/vllm#36702`: narrowed some AITER behavior to MI3xx and changed
  standard ROCm attention priorities. Preserve those decisions unless new
  validation justifies a targeted change.

Before opening a follow-up PR, search open PRs for `gfx1201`, `R9700`,
`RDNA4`, and the specific subsystem name. There has been active overlapping
ROCm work in attention, MoE, AITER, and FP8 feature gates.
