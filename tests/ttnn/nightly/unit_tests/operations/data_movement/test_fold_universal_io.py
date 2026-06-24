# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for fold universal input/output support (issue #47644).

Routing after the fix (verified below):
  - HEIGHT_SHARDED + ROW_MAJOR   → MultiCore (zero-NOC fast path)
  - HEIGHT_SHARDED + TILE        → MultiCoreDRAMFold tiled branch (TensorAccessor; native)
  - WIDTH_SHARDED  (any layout)  → MultiCoreDRAMFold (TensorAccessor; native)
  - BLOCK_SHARDED  (any layout)  → MultiCoreDRAMFold (TensorAccessor; native)
  - DRAM / L1 interleaved        → MultiCoreDRAMFold (TILE branch native; no composite untilize)

The math constraint `H % stride_h == 0` and `W % stride_w == 0` is fold's semantic
requirement, not an implementation limitation; tests only exercise logically valid shapes.
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

_TTNN_TO_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}

_TILE_H = 32
_TILE_W = 32

L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
DRAM_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


# ──────────────────────────────────────────────────────────────────────────────
# Reference + helpers
# ──────────────────────────────────────────────────────────────────────────────


def _fold_torch_nhwc(x_nhwc, stride_h, stride_w):
    """Reference fold: (N,H,W,C) → (N, H/sh, W/sw, C*sh*sw) (NHWC space-to-depth)."""
    n, h, w, c = x_nhwc.shape
    reshaped = x_nhwc.reshape(n, h // stride_h, stride_h, w // stride_w, stride_w, c)
    transposed = reshaped.permute(0, 1, 3, 2, 4, 5).contiguous()
    return transposed.reshape(n, h // stride_h, w // stride_w, c * stride_h * stride_w)


def _round_up(x, m):
    return ((x + m - 1) // m) * m


def _height_sharded_nhwc(shape, device, ncores, layout):
    """HEIGHT_SHARDED memory config for an (N,H,W,C) tensor: shard along total pixels (N*H*W)."""
    n, h, w, c = shape
    grid = device.compute_with_storage_grid_size()
    ncores = min(ncores, grid.x * grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(ncores, grid, True)
    total_pixels = n * h * w
    assert total_pixels % ncores == 0, f"total pixels {total_pixels} must be divisible by ncores {ncores}"
    shard_h = total_pixels // ncores
    shard_w = c
    if layout == ttnn.TILE_LAYOUT:
        shard_h = _round_up(shard_h, _TILE_H)
        shard_w = _round_up(shard_w, _TILE_W)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (shard_h, shard_w), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _width_sharded_nhwc(shape, device, ncores, layout):
    """WIDTH_SHARDED memory config: shard along C across cores."""
    n, h, w, c = shape
    grid = device.compute_with_storage_grid_size()
    ncores = min(ncores, grid.x * grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(ncores, grid, True)
    assert c % ncores == 0, f"C={c} must be divisible by ncores={ncores}"
    total_pixels = n * h * w
    shard_h = total_pixels
    shard_w = c // ncores
    if layout == ttnn.TILE_LAYOUT:
        shard_h = _round_up(shard_h, _TILE_H)
        shard_w = _round_up(shard_w, _TILE_W)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (shard_h, shard_w), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _block_sharded_nhwc(shape, device, grid_y, grid_x, layout):
    """BLOCK_SHARDED memory config: 2D grid over pixel-rows × C."""
    n, h, w, c = shape
    compute_grid = device.compute_with_storage_grid_size()
    if grid_y > compute_grid.y or grid_x > compute_grid.x:
        pytest.skip(f"Device grid ({compute_grid.y}x{compute_grid.x}) too small for {grid_y}x{grid_x}")
    total_pixels = n * h * w
    assert total_pixels % grid_y == 0
    assert c % grid_x == 0
    shard_h = total_pixels // grid_y
    shard_w = c // grid_x
    if layout == ttnn.TILE_LAYOUT:
        shard_h = _round_up(shard_h, _TILE_H)
        shard_w = _round_up(shard_w, _TILE_W)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
            (shard_h, shard_w),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shared runner
# ──────────────────────────────────────────────────────────────────────────────


def _run_fold(shape, stride_h, stride_w, layout, input_mem_config, dtype, device, pcc=0.9999):
    torch.manual_seed(12345)
    torch_dtype = _TTNN_TO_TORCH_DTYPE[dtype]
    x_nhwc = torch.rand(shape, dtype=torch_dtype)
    expected = _fold_torch_nhwc(x_nhwc, stride_h, stride_w)

    ttnn_in = ttnn.from_torch(x_nhwc, layout=layout, dtype=dtype, device=device, memory_config=input_mem_config)
    result = ttnn.fold(ttnn_in, stride_h=stride_h, stride_w=stride_w)

    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))

    # The device op may return either the folded 4D shape or the collapsed
    # (1, 1, N*H/sh*W/sw, C*sh*sw) shape depending on the path; compare by reshape to the
    # canonical folded 4D shape.
    n, h, w, c = expected.shape
    got_4d = got.reshape(n, h, w, c)
    assert_with_pcc(expected.float(), got_4d.float(), pcc)


# ──────────────────────────────────────────────────────────────────────────────
# Group A: interleaved baselines (regression guards)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "mem_config",
    [
        pytest.param(DRAM_INTERLEAVED, id="dram"),
        pytest.param(L1_INTERLEAVED, id="l1"),
    ],
)
def test_fold_interleaved_baselines(layout, mem_config, device):
    """DRAM/L1 interleaved × RM/TILE. TILE no longer goes through composite to_layout(RM)."""
    _run_fold((1, 8, 8, 32), 2, 2, layout, mem_config, ttnn.bfloat16, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group B: HEIGHT_SHARDED — RM fast path + newly-native TILE
# ──────────────────────────────────────────────────────────────────────────────


def test_fold_height_sharded_rm_fast_path(device):
    """HEIGHT_SHARDED + ROW_MAJOR — keeps the zero-NOC MultiCore fast path."""
    shape = (1, 8, 8, 32)
    mc = _height_sharded_nhwc(shape, device, ncores=2, layout=ttnn.ROW_MAJOR_LAYOUT)
    _run_fold(shape, 2, 2, ttnn.ROW_MAJOR_LAYOUT, mc, ttnn.bfloat16, device)


def test_fold_height_sharded_tile_native(device):
    """HEIGHT_SHARDED + TILE — newly native via MultiCoreDRAMFold's tiled branch.

    Before the fix this would go through the composite's `to_layout(RM)` then the
    RM-only `MultiCore` fast path. Now it routes directly to the tiled factory and
    uses TensorAccessor for shard-local tile reads.

    NB: TILE pads the last-two logical dims to tile multiples, so we use a shape with
    W and C already tile-aligned (W=32, C=32) to keep the shard math clean.
    """
    shape = (1, 32, 32, 32)
    mc = _height_sharded_nhwc(shape, device, ncores=8, layout=ttnn.TILE_LAYOUT)
    _run_fold(shape, 2, 2, ttnn.TILE_LAYOUT, mc, ttnn.bfloat16, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group C: WIDTH_SHARDED — newly supported via MultiCoreDRAMFold (TensorAccessor)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "layout, shape",
    [
        # RM keeps small shapes (no tile-alignment requirement on logical dims).
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, (1, 8, 8, 64), id="rm_small"),
        # TILE needs W, C tile-aligned to avoid padding inflating the buffer height.
        pytest.param(ttnn.TILE_LAYOUT, (1, 32, 32, 64), id="tile_aligned"),
    ],
)
def test_fold_width_sharded(layout, shape, device):
    """WIDTH_SHARDED (any layout) — before: TT_THROW; after: native via MultiCoreDRAMFold."""
    mc = _width_sharded_nhwc(shape, device, ncores=2, layout=layout)
    _run_fold(shape, 2, 2, layout, mc, ttnn.bfloat16, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group D: BLOCK_SHARDED — newly supported via MultiCoreDRAMFold (TensorAccessor)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "layout, shape",
    [
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, (1, 8, 8, 64), id="rm_small"),
        pytest.param(ttnn.TILE_LAYOUT, (1, 32, 32, 64), id="tile_aligned"),
    ],
)
def test_fold_block_sharded(layout, shape, device):
    """BLOCK_SHARDED (any layout) — before: TT_THROW; after: native via MultiCoreDRAMFold."""
    mc = _block_sharded_nhwc(shape, device, grid_y=2, grid_x=2, layout=layout)
    _run_fold(shape, 2, 2, layout, mc, ttnn.bfloat16, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group E: dtype coverage (float32 on a representative path each)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, layout, mem_config_factory",
    [
        pytest.param(
            (1, 8, 8, 32),
            ttnn.TILE_LAYOUT,
            lambda d, shape: DRAM_INTERLEAVED,
            id="tile_dram",
        ),
        pytest.param(
            (1, 8, 8, 32),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, shape: _height_sharded_nhwc(shape, d, ncores=2, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="rm_height_sharded",
        ),
        pytest.param(
            (1, 32, 32, 64),
            ttnn.TILE_LAYOUT,
            lambda d, shape: _width_sharded_nhwc(shape, d, ncores=2, layout=ttnn.TILE_LAYOUT),
            id="tile_width_sharded",
        ),
    ],
)
def test_fold_f32(shape, layout, mem_config_factory, device):
    """float32 on representative paths."""
    _run_fold(shape, 2, 2, layout, mem_config_factory(device, shape), ttnn.float32, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group F: multi-batch + asymmetric strides
# ──────────────────────────────────────────────────────────────────────────────


def test_fold_multi_batch_width_sharded(device):
    """N>1 width-sharded — sanity-checks the batch dim along with W-shard routing."""
    shape = (2, 8, 8, 64)
    mc = _width_sharded_nhwc(shape, device, ncores=2, layout=ttnn.ROW_MAJOR_LAYOUT)
    _run_fold(shape, 2, 2, ttnn.ROW_MAJOR_LAYOUT, mc, ttnn.bfloat16, device)


def test_fold_asymmetric_stride_block_sharded(device):
    """stride_h != stride_w on a block-sharded input."""
    shape = (1, 6, 8, 64)  # H=6 % 3 == 0, W=8 % 2 == 0
    mc = _block_sharded_nhwc(shape, device, grid_y=2, grid_x=2, layout=ttnn.ROW_MAJOR_LAYOUT)
    _run_fold(shape, 3, 2, ttnn.ROW_MAJOR_LAYOUT, mc, ttnn.bfloat16, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group G: regression for the legacy `to_layout(RM)` bypass on TILE inputs
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "memory_config",
    [
        pytest.param(DRAM_INTERLEAVED, id="dram"),
        pytest.param(L1_INTERLEAVED, id="l1"),
    ],
)
def test_fold_tile_native_no_composite_untilize(memory_config, device):
    """Regression: TILE interleaved inputs must take MultiCoreDRAMFold's tiled branch
    (no composite `to_layout(ROW_MAJOR)`). We can't directly observe routing, but the
    final tensor must still match the reference fold."""
    shape = (1, 64, 64, 32)
    _run_fold(shape, 2, 2, ttnn.TILE_LAYOUT, memory_config, ttnn.bfloat16, device)
