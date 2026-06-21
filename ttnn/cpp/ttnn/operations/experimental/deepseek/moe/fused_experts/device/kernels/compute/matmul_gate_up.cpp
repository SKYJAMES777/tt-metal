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

// Per-core gate_up matmul + SwiGLU gate + down matmul (runs on every compute core).
//
// PHASE 1 (SwiGLU cores only -- those owning a slice of the I dim):
//   Each SwiGLU core owns a 2-tile (64 column) slice of the SwiGLU output I dimension. Its
//   gate_up weight shard ([K, 128] == [k_tiles, 4] tiles) holds the gate columns (tile cols
//   0,1) and paired up columns (tile cols 2,3) for that slice. For every selected expert:
//       gate = x @ gate_w   (cb_weights tile cols 0,1) -> [32, 64]
//       up   = x @ up_w     (cb_weights tile cols 2,3) -> [32, 64]
//       cb_out = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit) -> [32, 64]
//   where x (cb_input) is resident as Kt == H/32 activation tiles. cb_out (this core's 2-tile
//   slice of act[1, I]) is scattered by the writer to core {0,0}, gathered into the full
//   activation, and broadcast back into every core's cb_act.
//
// PHASE 2 (all cores): the down matmul. cb_act holds the full activation act[1, I] (i_tiles
//   tiles, K order). Each core multiplies it by its down weight shard ([I, H/64] ==
//   [i_tiles, 2] tiles) to produce its 2-tile (64 column) slice of the output row[1, H]:
//       cb_down_out = act @ down_w   (cb_down_w tile (k, n) at k*2 + n) -> [32, 64]
//   summed over k == 0..i_tiles-1. cb_down_out feeds the writer (DRAM output [num_active, 1, H]).
//
// Compile-time args:
//   0: num_active   (routing-selected experts to run)
//   1: k_tiles      (H / 32, gate_up contraction)
//   2: i_tiles      (I / 32; SwiGLU output cols AND down contraction (act K-tiles))
//   3: cb_input     (activation tiles)
//   4: cb_weights   (this core's per-expert [K, 128] gate+up slice)
//   5: cb_mm        (gate_up matmul staging: 4 tiles = gate 0,1 | up 2,3)
//   6: cb_out       (this core's 2 SwiGLU output tiles per expert)
//   7: limit_bits   (SwiGLU clamp limit as a float bit pattern)
//   8: cb_act       (full gathered activation act[1, I], i_tiles tiles)
//   9: cb_down_w    (this core's per-expert [I, 64] down slice = i_tiles*2 tiles)
//  10: cb_down_out  (this core's 2 down output tiles per expert)
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
    constexpr uint32_t cb_act_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_down_w_id = get_compile_time_arg_val(9);
    constexpr uint32_t cb_down_out_id = get_compile_time_arg_val(10);

    const uint32_t col_start_tile = get_arg_val<uint32_t>(0);
    const bool swiglu_core = col_start_tile < i_tiles;

    constexpr uint32_t kOutTilesPerCore = 2;
    constexpr uint32_t kShardTileCols = 2 * kOutTilesPerCore;  // gate 2 | up 2
    const uint32_t slice_tiles = k_tiles * kShardTileCols;
    // down weight shard: [I, 64] == [i_tiles, 2] tiles (full K = I, this core's 2 H-cols).
    const uint32_t down_slice_tiles = i_tiles * kOutTilesPerCore;

    // gate: clamp(min = -inf, max = limit); up: clamp(min = -limit, max = limit).
    constexpr uint32_t kNegInfBits = 0xFF800000u;
    constexpr uint32_t neg_limit_bits = limit_bits ^ 0x80000000u;

    CircularBuffer in_cb(cb_input_id);
    CircularBuffer w_cb(cb_weights_id);
    CircularBuffer mm_cb(cb_mm_id);
    CircularBuffer out_cb(cb_out_id);
    CircularBuffer act_cb(cb_act_id);
    CircularBuffer down_w_cb(cb_down_w_id);
    CircularBuffer down_out_cb(cb_down_out_id);

    mm_init(cb_input_id, cb_weights_id, cb_mm_id);

    // Activation x is broadcast once and reused for every expert's gate_up matmul.
    if (swiglu_core) {
        in_cb.wait_front(k_tiles);
    }

    for (uint32_t e = 0; e < num_active; ++e) {
        if (swiglu_core) {
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

        // ---- down matmul (all cores): cb_act @ cb_down_w -> cb_down_out (2 tiles). ----
        act_cb.wait_front(i_tiles);
        down_w_cb.wait_front(down_slice_tiles);

        mm_init_short(cb_act_id, cb_down_w_id);
        reconfig_data_format(cb_down_w_id, cb_act_id);
        pack_reconfig_data_format(cb_down_out_id);
        down_out_cb.reserve_back(kOutTilesPerCore);

        tile_regs_acquire();
        for (uint32_t n = 0; n < kOutTilesPerCore; ++n) {
            for (uint32_t k = 0; k < i_tiles; ++k) {
                matmul_tiles(cb_act_id, cb_down_w_id, k, k * kOutTilesPerCore + n, n);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_down_out_id);
        pack_tile(1, cb_down_out_id);
        tile_regs_release();

        down_out_cb.push_back(kOutTilesPerCore);
        act_cb.pop_front(i_tiles);
        down_w_cb.pop_front(down_slice_tiles);
    }

    if (swiglu_core) {
        in_cb.pop_front(k_tiles);
    }
}
