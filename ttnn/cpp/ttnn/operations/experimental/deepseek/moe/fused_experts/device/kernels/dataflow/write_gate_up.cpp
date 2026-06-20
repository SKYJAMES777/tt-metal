// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Writer kernel (runs on every compute core).
//
// Pushes this core's gate_up + SwiGLU activation to DRAM. The output tensor is
// [num_active, 1, I] in TILE layout (the decode token row padded to a 32-row tile),
// so in tiles it is [num_active, 1, i_tiles] with i_tiles == I / 32. Each active core
// owns 2 output tiles (its 64-column I slice): the I-dim tiles [col_start_tile,
// col_start_tile + 1]. For the i-th selected expert they live at output pages
// i * i_tiles + col_start_tile + {0, 1}.
//
// Consumes cb_out (2 tiles per expert) produced by the compute kernel, one expert at
// a time, for the num_active selected experts.
//
// Compile-time args:
//   0: num_active     (routing-selected experts to run)
//   1: i_tiles        (I / 32; output page stride per expert row and idle guard)
//   2: cb_out         (this core's 2 activation tiles per expert)
//   3: out_tile_bytes (bytes per output tile)
//   4+: TensorAccessorArgs(output)
//
// Runtime args:
//   0: output base address
//   1: col_start_tile (this core's first SwiGLU output tile = compute_index * 2)
void kernel_main() {
    constexpr uint32_t num_active = get_compile_time_arg_val(0);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(2);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(3);

    constexpr auto out_args = TensorAccessorArgs<4>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t col_start_tile = get_arg_val<uint32_t>(1);
    if (col_start_tile >= i_tiles) {
        return;  // this core owns no output slice
    }

    constexpr uint32_t kOutTiles = 2;

    Noc noc;
    const auto out = TensorAccessor(out_args, out_addr);
    CircularBuffer cb_out(cb_out_id);

    for (uint32_t e = 0; e < num_active; ++e) {
        cb_out.wait_front(kOutTiles);
        for (uint32_t t = 0; t < kOutTiles; ++t) {
            const uint32_t page = e * i_tiles + col_start_tile + t;
            noc.async_write(cb_out, out, out_tile_bytes, {.offset_bytes = t * out_tile_bytes}, {.page_id = page});
        }
        noc.async_write_barrier();
        cb_out.pop_front(kOutTiles);
    }
}
