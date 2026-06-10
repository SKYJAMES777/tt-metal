// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    DataflowBuffer dfb_in(dfb::in_data);
    DataflowBuffer dfb_in_scaler(dfb::in_scaler);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in_data, dfb::in_scaler, dfb::out);
    tilizeA_B_reduce_init<true /*neginf_srcA*/, false /*zero_srcA_reduce*/>(
        dfb::in_data, dfb::in_scaler, per_core_block_tile_cnt, dfb::out);

    dfb_in_scaler.wait_front(1);

    tile_regs_acquire();
    tile_regs_wait();
    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        dfb_in.wait_front(per_core_block_tile_cnt);
        unpack_tilizeA_B_block<true /*neginf_srcA*/, true /*reload_srcB*/, false, false>(
            dfb::in_data,
            dfb::in_scaler,
            per_core_block_tile_cnt,
            0 /*tile idx for Src b is 0 because only 1 scaler tile is loaded*/);
        for (uint32_t i = 0; i < per_core_block_tile_cnt; ++i) {
            reduce_tile_math<REDUCE_OP, REDUCE_DIM>(i);
        }
        dfb_in.pop_front(per_core_block_tile_cnt);
    }
    for (uint32_t i = 0; i < per_core_block_tile_cnt; ++i) {
        dfb_out.reserve_back(1);
        pack_tile(i, dfb::out);
        dfb_out.push_back(1);
    }
    tile_regs_commit();
    tile_regs_release();

    reduce_uninit();
}
