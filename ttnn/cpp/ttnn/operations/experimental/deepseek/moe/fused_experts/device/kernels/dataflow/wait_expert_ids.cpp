// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

#include "fetch_gate_up.h"

// Reader kernel (runs on every compute core except the two senders {0,0} and {1,0}).
//
// Waits for both broadcasts to land in this core's L1:
//   - {1,0} multicasts the activation row into cb_input and bumps sem_input_id.
//   - {0,0} multicasts the expert ids into cb_bcast and bumps sem_id.
// It then publishes the activation to the compute kernel (cb_input) and fetches
// this core's gate_up weight slice for the routing-selected experts (cb_weights).
//
// Compile-time args:
//   0: sem_id        (expert-ids-ready / sequencing semaphore)
//   1: sem_input_id  (input-ready semaphore)
//   2: num_active    (routing-selected experts to run)
//   3: cb_input      (activation tiles, published to compute)
//   4: cb_weights    (this core's per-expert weight slice)
//   5: k_tiles       (H / 32)
//   6: i_tiles       (I / 32, SwiGLU output tile cols)
//   7: tile_bytes
//   8: cb_bcast      (broadcast hit-expert ids, read by the weight fetch)
//   9+: TensorAccessorArgs(gate_up)
//
// Runtime args:
//   0: col_start_tile  (this core's first SwiGLU output tile)
//   1+: gate_up base addresses (one per expert)
void kernel_main() {
    constexpr uint32_t sem_id = get_compile_time_arg_val(0);
    constexpr uint32_t sem_input_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_active = get_compile_time_arg_val(2);
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(4);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t cb_bcast_id = get_compile_time_arg_val(8);

    constexpr auto gate_up_args = TensorAccessorArgs<9>();

    const uint32_t col_start_tile = get_arg_val<uint32_t>(0);
    constexpr uint32_t kWeightAddrBase = 1;

    // Activation arrived via multicast: publish it to the compute kernel.
    Semaphore<>(sem_input_id).wait(1);
    publish_input(cb_input_id, k_tiles);

    // Expert ids arrived, then fetch our selected weight slices.
    Semaphore<>(sem_id).wait(1);
    Noc noc;
    fetch_gate_up_slices(
        noc,
        cb_bcast_id,
        cb_weights_id,
        num_active,
        k_tiles,
        i_tiles,
        tile_bytes,
        col_start_tile,
        gate_up_args,
        kWeightAddrBase);
}
