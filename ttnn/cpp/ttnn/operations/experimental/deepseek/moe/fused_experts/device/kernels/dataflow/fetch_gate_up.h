// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"

// Per-core gate_up weight fetch (shared by all kernels in this op).
//
// Each active core owns a 2-tile (64 column) slice of the SwiGLU output I dimension:
// output tiles [col_start_tile, col_start_tile + 1]. Computing that SwiGLU slice needs
// both the gate columns [64c, 64c+64) and the paired up columns [I+64c, I+64c+64) of
// the [K, 2I] gate_up weight (c == this core's index).
//
// To keep all of that in one DRAM shard, the gate_up weight is reshaped+permuted on
// the host into per-core [gate_64 | up_64] blocks, so each shard is this core's
// [K, 128] slice (shard shape [K, 128] in elements -> [k_tiles, 4] in tiles: tile cols
// 0,1 == gate, tile cols 2,3 == up). The shards are round-robin distributed across the
// DRAM banks with contiguous pages, so this core's entire gate+up weight for one
// expert is pulled in a *single* NoC read. Shard id == this core's index ==
// col_start_tile / 2.
//
// The op takes *all* experts' weights as input and uses the routing weights to
// select which experts to run. The selected ("hit") expert ids are computed on
// device and broadcast into cb_bcast as a compacted, ascending list; this core
// fetches only those `num_active` experts. The i-th fetched shard belongs to
// expert ids[i], and feeds the i-th output row (matmul + writer loop in lock-step).
//
// Arguments:
//   noc               NoC instance to use for the reads.
//   cb_bcast_id       CB holding the broadcast hit-expert ids (ascending).
//   cb_weights_id     CB receiving this core's weight slice (producer side).
//   num_active        Number of routing-selected experts to run.
//   k_tiles           K / 32 (number of tile rows of the weight).
//   i_tiles           I / 32 (SwiGLU output tile cols; cores past it are idle).
//   tile_bytes        Size of one tile in bytes.
//   col_start_tile    This core's first SwiGLU output tile (= compute_index * 2).
//   gate_up_args      Shared TensorAccessorArgs (all experts share one layout).
//   rt_w_addr_base    Runtime-arg index of the first gate_up base address.
// The activation row is delivered into every core's cb_input L1 region by the
// input broadcaster's multicast (receivers) or by a direct DRAM read (the
// broadcaster itself). Advancing cb_input by k_tiles pages publishes it to the
// matmul compute kernel. Receivers call this once the input-ready semaphore fires.
inline void publish_input(uint32_t cb_input_id, uint32_t k_tiles) {
    CircularBuffer cb_input(cb_input_id);
    cb_input.reserve_back(k_tiles);
    cb_input.push_back(k_tiles);
}

template <typename GateUpArgs>
void fetch_gate_up_slices(
    const Noc& noc,
    uint32_t cb_bcast_id,
    uint32_t cb_weights_id,
    uint32_t num_active,
    uint32_t k_tiles,
    uint32_t i_tiles,
    uint32_t tile_bytes,
    uint32_t col_start_tile,
    const GateUpArgs& gate_up_args,
    uint32_t rt_w_addr_base) {
    // Cores whose SwiGLU output slice falls outside I do nothing.
    if (col_start_tile >= i_tiles) {
        return;
    }

    // The shard is this core's 2 output tiles' gate (tiles 0,1) + up (tiles 2,3) = 4 tile cols.
    constexpr uint32_t kOutTilesPerCore = 2;
    constexpr uint32_t kShardTileCols = 2 * kOutTilesPerCore;  // gate 2 | up 2
    const uint32_t slice_tiles = k_tiles * kShardTileCols;
    // One DRAM shard == this core's whole [K, 128] gate+up slice; read it in one shot.
    const uint32_t slice_bytes = slice_tiles * tile_bytes;
    const uint32_t shard_id = col_start_tile / kOutTilesPerCore;

    // The hit-expert ids were broadcast into cb_bcast (ascending, compacted at the
    // front). cb_bcast is never advanced, so the ids live at its write pointer.
    CircularBuffer cb_bcast(cb_bcast_id);
    CoreLocalMem<volatile uint32_t> ids(cb_bcast.get_write_ptr());

    CircularBuffer cb_weights(cb_weights_id);

    for (uint32_t i = 0; i < num_active; ++i) {
        const uint32_t expert = ids[i];
        const uint32_t w_addr = get_arg_val<uint32_t>(rt_w_addr_base + expert);
        const auto w = TensorAccessor(gate_up_args, w_addr);

        // Single NoC read of this expert's entire shard for this core.
        cb_weights.reserve_back(slice_tiles);
        ShardView w_shard(w);
        noc.async_read(w_shard, cb_weights, slice_bytes, {.shard_id = shard_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_weights.push_back(slice_tiles);
    }
}
