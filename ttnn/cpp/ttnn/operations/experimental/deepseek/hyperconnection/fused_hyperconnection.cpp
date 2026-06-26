// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_hyperconnection.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::experimental::deepseek::hyperconnection {

std::tuple<Tensor, Tensor, Tensor> fused_hyperconnection(
    const Tensor& hidden_streams,
    const Tensor& pre_w,
    const Tensor& post_w,
    const Tensor& comb_w,
    const Tensor& pre_bias,
    const Tensor& post_bias,
    const Tensor& comb_bias,
    uint32_t num_streams,
    uint32_t sinkhorn_iters,
    float pre_scale,
    float post_scale,
    float comb_scale,
    float eps,
    const std::optional<MemoryConfig>& memory_config) {
    const auto& shape = hidden_streams.logical_shape();
    const uint32_t b = static_cast<uint32_t>(shape[0]);
    const uint32_t s = static_cast<uint32_t>(shape[1]);
    const uint32_t hc = num_streams;
    const uint32_t d = static_cast<uint32_t>(shape[-1]);
    const uint32_t t = b * s;

    // pre = sigmoid(pre_w * pre_scale + pre_bias) + eps.
    Tensor pre = ttnn::add(ttnn::sigmoid(ttnn::add(ttnn::multiply(pre_w, pre_scale), pre_bias)), eps);
    // post = 2 * sigmoid(post_w * post_scale + post_bias).
    Tensor post = ttnn::multiply(ttnn::sigmoid(ttnn::add(ttnn::multiply(post_w, post_scale), post_bias)), 2.0f);

    // comb logits -> [1,T,H,H]; softmax over last dim, then Sinkhorn (alternate row/col
    // normalisation) onto the doubly-stochastic manifold.
    Tensor comb_logits = ttnn::add(ttnn::multiply(comb_w, comb_scale), comb_bias);  // [1,1,T,H*H]
    comb_logits = ttnn::reshape(comb_logits, ttnn::Shape({1, t, hc, hc}));
    Tensor comb = ttnn::add(ttnn::softmax(comb_logits, -1), eps);
    comb = ttnn::divide(comb, ttnn::add(ttnn::sum(comb, /*dim=*/-2, /*keepdim=*/true), eps));  // column
    for (uint32_t i = 1; i < sinkhorn_iters; ++i) {
        comb = ttnn::divide(comb, ttnn::add(ttnn::sum(comb, /*dim=*/-1, /*keepdim=*/true), eps));  // row
        comb = ttnn::divide(comb, ttnn::add(ttnn::sum(comb, /*dim=*/-2, /*keepdim=*/true), eps));  // column
    }

    // collapsed = sum_h pre[..,h] * hidden_streams[..,h,:]  (weighted stream sum).
    Tensor hs = ttnn::reshape(hidden_streams, ttnn::Shape({1, t, hc, d}));
    Tensor pre_col = ttnn::reshape(pre, ttnn::Shape({1, t, hc, 1}));
    Tensor collapsed = ttnn::sum(ttnn::multiply(hs, pre_col), /*dim=*/-2, /*keepdim=*/true);  // [1,T,1,D]

    post = ttnn::reshape(post, ttnn::Shape({b, s, hc, 1}));
    comb = ttnn::reshape(comb, ttnn::Shape({b, s, hc, hc}));
    collapsed = ttnn::reshape(collapsed, ttnn::Shape({b, s, 1, d}));

    if (memory_config.has_value()) {
        post = ttnn::to_memory_config(post, *memory_config);
        comb = ttnn::to_memory_config(comb, *memory_config);
        collapsed = ttnn::to_memory_config(collapsed, *memory_config);
    }
    return {post, comb, collapsed};
}

}  // namespace ttnn::experimental::deepseek::hyperconnection
