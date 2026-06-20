// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"

// Per-core gate_up matmul + SwiGLU gate (runs on every compute core).
//
// Each active core owns a 2-tile (64 column) slice of the SwiGLU output I dimension.
// Its gate_up weight shard ([K, 128] == [k_tiles, 4] tiles) holds the gate columns
// (tile cols 0,1) and the paired up columns (tile cols 2,3) for that slice. For every
// routing-selected expert it computes, for each of its 2 output tiles:
//     gate = x @ gate_w   (cb_weights tile cols 0,1) -> [32, 64]
//     up   = x @ up_w     (cb_weights tile cols 2,3) -> [32, 64]
//     out  = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit) -> [32, 64]
// where x (cb_input) is resident as Kt == H/32 activation tiles (one tile-row,
// M == 1 padded to 32). The matmul results are staged in cb_mm (4 tiles: gate 0,1 |
// up 2,3) and the SwiGLU output (2 tiles) is pushed to cb_out for the writer. Output
// row i feeds the i-th routing-selected expert.
//
// Compile-time args:
//   0: num_active  (routing-selected experts to run)
//   1: k_tiles     (H / 32)
//   2: i_tiles     (I / 32)  -- used to skip cores outside the SwiGLU output
//   3: cb_input    (activation tiles)
//   4: cb_weights  (this core's per-expert [K, 128] gate+up slice)
//   5: cb_mm       (matmul staging: 4 tiles = gate 0,1 | up 2,3)
//   6: cb_out      (this core's 2 SwiGLU output tiles per expert)
//   7: limit_bits  (SwiGLU clamp limit as a float bit pattern)
//
// Runtime args:
//   0: col_start_tile (this core's first SwiGLU output tile = compute_index * 2)
void kernel_main() {
    constexpr uint32_t num_active = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_mm_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(6);
    constexpr uint32_t limit_bits = get_compile_time_arg_val(7);

    const uint32_t col_start_tile = get_arg_val<uint32_t>(0);
    if (col_start_tile >= i_tiles) {
        return;  // this core owns no SwiGLU output slice
    }

    constexpr uint32_t kOutTilesPerCore = 2;
    constexpr uint32_t kShardTileCols = 2 * kOutTilesPerCore;  // gate 2 | up 2
    const uint32_t slice_tiles = k_tiles * kShardTileCols;

    // gate: clamp(min = -inf, max = limit); up: clamp(min = -limit, max = limit).
    constexpr uint32_t kNegInfBits = 0xFF800000u;
    constexpr uint32_t neg_limit_bits = limit_bits ^ 0x80000000u;

    CircularBuffer in_cb(cb_input_id);
    CircularBuffer w_cb(cb_weights_id);
    CircularBuffer mm_cb(cb_mm_id);
    CircularBuffer out_cb(cb_out_id);

    mm_init(cb_input_id, cb_weights_id, cb_mm_id);

    // Activation is broadcast once and reused for every expert.
    in_cb.wait_front(k_tiles);

    for (uint32_t e = 0; e < num_active; ++e) {
        w_cb.wait_front(slice_tiles);

        // ---- gate + up matmul -> cb_mm (gate tiles 0,1; up tiles 2,3). ----
        mm_init_short(cb_input_id, cb_weights_id);
        reconfig_data_format(cb_weights_id, cb_input_id);
        pack_reconfig_data_format(cb_mm_id);
        mm_cb.reserve_back(2 * kOutTilesPerCore);

        // gate (weight tile (k, n) at k*4 + n) -> dst 0,1 -> cb_mm 0,1
        tile_regs_acquire();
        for (uint32_t n = 0; n < kOutTilesPerCore; ++n) {
            for (uint32_t k = 0; k < k_tiles; ++k) {
                matmul_tiles(cb_input_id, cb_weights_id, k, k * kShardTileCols + n, n);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_mm_id, 0);
        pack_tile<true>(1, cb_mm_id, 1);
        tile_regs_release();

        // up (weight tile (k, n) at k*4 + 2 + n) -> dst 0,1 -> cb_mm 2,3
        tile_regs_acquire();
        for (uint32_t n = 0; n < kOutTilesPerCore; ++n) {
            for (uint32_t k = 0; k < k_tiles; ++k) {
                matmul_tiles(cb_input_id, cb_weights_id, k, k * kShardTileCols + kOutTilesPerCore + n, n);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_mm_id, 2);
        pack_tile<true>(1, cb_mm_id, 3);
        tile_regs_release();

        mm_cb.push_back(2 * kOutTilesPerCore);
        w_cb.pop_front(slice_tiles);

        // ---- SwiGLU: cb_mm (gate 0,1 | up 2,3) -> cb_out (2 tiles). ----
        mm_cb.wait_front(2 * kOutTilesPerCore);
        copy_tile_to_dst_init_short(cb_mm_id);
        reconfig_data_format_srca(cb_mm_id);
        pack_reconfig_data_format(cb_out_id);
        out_cb.reserve_back(kOutTilesPerCore);

        tile_regs_acquire();
        copy_tile(cb_mm_id, 0, 0);  // gate 0 -> dst 0
        copy_tile(cb_mm_id, 1, 1);  // gate 1 -> dst 1
        copy_tile(cb_mm_id, 2, 2);  // up 0   -> dst 2
        copy_tile(cb_mm_id, 3, 3);  // up 1   -> dst 3

        // gate = silu(clamp(gate, max = limit))
        clamp_tile_init();
        clamp_tile(0, kNegInfBits, limit_bits);
        clamp_tile(1, kNegInfBits, limit_bits);
        silu_tile_init();
        silu_tile(0);
        silu_tile(1);

        // up = clamp(up, -limit, limit)
        clamp_tile_init();
        clamp_tile(2, neg_limit_bits, limit_bits);
        clamp_tile(3, neg_limit_bits, limit_bits);

        // out = gate * up
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 0);
        mul_binary_tile(1, 3, 1);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out_id);
        pack_tile(1, cb_out_id);
        tile_regs_release();

        mm_cb.pop_front(2 * kOutTilesPerCore);
        out_cb.push_back(kOutTilesPerCore);
    }

    in_cb.pop_front(k_tiles);
}
