// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"

// Writer / gather-scatter kernel (runs on every compute core), structured in two phases:
//
//   PHASE 1 -- GATHER (SwiGLU cores only): for every expert, copies this core's 2-tile
//      SwiGLU activation slice (cb_out) into core {0,0}'s cb_act at tile offset
//      (e*i_tiles + col_start_tile) -- a single NoC write to the leader -- then bumps the
//      leader's gather semaphore. After all experts' chunks from all SwiGLU cores have
//      landed, {0,0}'s cb_act holds the whole [num_active, I] activation block, which it
//      broadcasts back to every core in one shot (see the reader kernels).
//
//   PHASE 2 -- WRITE DOWN OUTPUT (all cores): for every expert, writes this core's 2-tile
//      slice of the down matmul output (cb_down_out) to the [num_active, 1, H] DRAM output.
//      No semaphore is needed -- cb_act is filled once and never reused, so there is no
//      activation-free back-pressure.
//
// Down output is TILE [num_active, 1, H] (the decode token row padded to a 32-row tile),
// so in tiles it is [num_active, 1, h_tiles] with h_tiles == H/32. Core idx owns the 2
// H-dim tiles [col_start_tile, col_start_tile + 1]; for expert e they live at output pages
// e * h_tiles + col_start_tile + {0, 1}.
//
// Compile-time args:
//   0: num_active     (routing-selected experts to run)
//   1: i_tiles        (I/32; SwiGLU-core guard for the gather scatter + activation stride)
//   2: h_tiles        (H/32; output page stride per expert row)
//   3: cb_out         (this core's 2 SwiGLU activation tiles per expert)
//   4: cb_down_out    (this core's 2 down output tiles per expert)
//   5: cb_act         (gathered activation; used only to locate the leader's L1 address)
//   6: act_tile_bytes (bytes per activation tile)
//   7: out_tile_bytes (bytes per output tile)
//   8: sem_gather     (leader's gather semaphore)
//   9+: TensorAccessorArgs(output)
//
// Runtime args:
//   0: output base address
//   1: col_start_tile (this core's first output tile = compute_index * 2)
//   2: leader_noc_x   3: leader_noc_y (core {0,0} NoC coords for this writer's NoC)
void kernel_main() {
    constexpr uint32_t num_active = get_compile_time_arg_val(0);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t h_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_down_out_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_act_id = get_compile_time_arg_val(5);
    constexpr uint32_t act_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t sem_gather_id = get_compile_time_arg_val(8);

    constexpr auto out_args = TensorAccessorArgs<9>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t col_start_tile = get_arg_val<uint32_t>(1);
    const uint32_t leader_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t leader_noc_y = get_arg_val<uint32_t>(3);

    constexpr uint32_t kOutTiles = 2;
    const bool swiglu_core = col_start_tile < i_tiles;

    Noc noc;
    const auto out = TensorAccessor(out_args, out_addr);
    CircularBuffer cb_out(cb_out_id);
    CircularBuffer cb_down_out(cb_down_out_id);
    CircularBuffer cb_act(cb_act_id);

    // cb_act is allocated identically on every core, so the leader's cb_act base equals this
    // core's cb_act write pointer (the gather sink is never popped before the broadcast).
    const uint32_t leader_act_base = cb_act.get_write_ptr();

    Semaphore<> sem_gather(sem_gather_id);

    // ---- Phase 1: scatter every expert's SwiGLU slice into the leader's cb_act. ----
    if (swiglu_core) {
        for (uint32_t e = 0; e < num_active; ++e) {
            cb_out.wait_front(kOutTiles);
            const uint32_t leader_addr = leader_act_base + (e * i_tiles + col_start_tile) * act_tile_bytes;
            noc.async_write(
                cb_out,
                UnicastEndpoint{},
                kOutTiles * act_tile_bytes,
                {.offset_bytes = 0},
                {.noc_x = leader_noc_x, .noc_y = leader_noc_y, .addr = leader_addr});
            noc.async_write_barrier();
            sem_gather.up(noc, leader_noc_x, leader_noc_y, 1);
            cb_out.pop_front(kOutTiles);
        }
    }

    // ---- Phase 2: write every expert's down output slice to DRAM. ----
    for (uint32_t e = 0; e < num_active; ++e) {
        cb_down_out.wait_front(kOutTiles);
        for (uint32_t t = 0; t < kOutTiles; ++t) {
            const uint32_t page = e * h_tiles + col_start_tile + t;
            noc.async_write(cb_down_out, out, out_tile_bytes, {.offset_bytes = t * out_tile_bytes}, {.page_id = page});
        }
        noc.async_write_barrier();
        cb_down_out.pop_front(kOutTiles);
    }
}
