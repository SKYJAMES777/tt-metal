// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"

/**
 * @brief Determines whether a reduce operation should use the SFPU max path.
 *
 * Int32 MAX on REDUCE_ROW/COL uses SFPU. Int32 MAX REDUCE_SCALAR is unsupported.
 */
template <ckernel::PoolType pool_type, ckernel::ReduceDim reduce_dim, DataFormat data_format>
constexpr bool is_sfpu_reduce_path() {
    if constexpr (pool_type != ckernel::PoolType::MAX) {
        return false;
    }
    if constexpr (data_format != DataFormat::Int32) {
        return false;
    }
    if constexpr (reduce_dim == ckernel::ReduceDim::REDUCE_SCALAR) {
        return false;
    }
    return reduce_dim == ckernel::ReduceDim::REDUCE_ROW || reduce_dim == ckernel::ReduceDim::REDUCE_COL;
}
