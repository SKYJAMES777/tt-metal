// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_device_operation.hpp"

#include <algorithm>
#include <bit>
#include <string>

#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

using namespace tt;
using namespace tt::tt_metal;

namespace {
constexpr std::string_view kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels";

// Compute grid for the broadcast / matmul: 8x8 = 64 cores.
constexpr uint32_t GRID_X = 8;
constexpr uint32_t GRID_Y = 8;

// Each active core owns kTilesPerCore SwiGLU output tiles (64 columns). Its gate_up
// shard is 2*kTilesPerCore tiles wide ([gate_64 | up_64]) and its first output tile
// (== col_start_tile) is compute_index * kTilesPerCore.
constexpr uint32_t kTilesPerCore = 2;

uint32_t align_up_32(uint32_t x) { return (x + 31u) & ~31u; }
}  // namespace

// Pipeline:
//   - {0,0} (NoC 0) reads routing weights, computes/broadcasts the selected ("hit")
//     expert ids (ascending), and fetches its gate_up slices for those experts.
//   - {1,0} (NoC 1) reads the decode activation row and broadcasts it to every
//     core's L1 (cb_input), then fetches its gate_up slices.
//   - Each active core fetches its [K, 128] gate_up shard (one NoC read each) -- the
//     per-core [gate_64 | up_64] block holding its 2 gate tiles and their paired up
//     tiles -- for each of the num_active selected experts. The compute kernel runs the
//     gate_up matmul and the fused SwiGLU activation, producing this core's 2 output
//     tiles per expert, and the writer writes them to the [num_active, 1, I] DRAM
//     output, one output row per selected expert.
ProgramDescriptor FusedExpertsDeviceOperation::MultiCore::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& routing_weights = tensor_args.routing_weights;
    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* routing_buffer = routing_weights.buffer();
    auto* input_buffer = input_tensor.buffer();
    auto* out_buffer = output_tensor.buffer();
    auto* device = routing_weights.device();

    const auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(
        grid.x >= GRID_X && grid.y >= GRID_Y,
        "fused_experts: expected at least {}x{} compute grid, got {}x{}",
        GRID_X,
        GRID_Y,
        grid.x,
        grid.y);

    // The op takes all experts' weights and uses the routing weights to select which
    // ones to run. `num_weights` is the total provided (and the routing-row width);
    // `num_active` is the routing-selected count that drives the fetch / compute /
    // writer loops and the number of output rows.
    const uint32_t num_weights = static_cast<uint32_t>(tensor_args.gate_up_weights.size());
    const uint32_t num_active = operation_attributes.num_experts;
    const uint32_t sentinel = num_weights;  // "no expert" marker for unused id slots

    // gate_up weights are [K=H, N=2I] per expert (TILE layout), reshaped+permuted on the
    // host into per-core [gate_64 | up_64] blocks. Each core fetches its [K, 128]
    // (k_tiles x 4-tile) column slice -- one DRAM shard (gate cols 0,1 | up cols 2,3) --
    // for every selected expert. All experts share the same layout, so one
    // TensorAccessorArgs (from weight 0) is reused.
    const auto& gate_up0 = tensor_args.gate_up_weights.front();
    auto* gate_up0_buffer = gate_up0.buffer();
    constexpr uint32_t TILE_DIM = 32;
    const uint32_t k_tiles = static_cast<uint32_t>(gate_up0.logical_shape()[-2]) / TILE_DIM;
    const uint32_t n_tiles = static_cast<uint32_t>(gate_up0.logical_shape()[-1]) / TILE_DIM;  // 2I / 32
    const uint32_t i_tiles = n_tiles / 2u;  // SwiGLU output tile cols (I / 32)
    const uint32_t weight_tile_bytes = static_cast<uint32_t>(gate_up0_buffer->page_size());
    // Each core's weight slice is its 2 gate tiles + 2 paired up tiles per k-row.
    const uint32_t weight_slice_tiles = k_tiles * (2u * kTilesPerCore);
    // Double-buffer the weight slice so the reader can prefetch the next expert.
    const uint32_t weights_cb_bytes = 2u * weight_slice_tiles * weight_tile_bytes;

    const tt::DataFormat gate_up_df = datatype_to_dataformat_converter(gate_up0.dtype());
    const tt::DataFormat routing_df = datatype_to_dataformat_converter(routing_weights.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output_tensor.dtype());
    const tt::DataFormat input_df = datatype_to_dataformat_converter(input_tensor.dtype());

    constexpr uint32_t routing_elem_bytes = 2;  // bfloat16
    constexpr uint32_t out_elem_bytes = 4;      // uint32 expert ids (broadcast scratch)
    // Routing row and the id broadcast span all provided experts.
    const uint32_t routing_page_bytes = num_weights * routing_elem_bytes;
    const uint32_t bcast_page_bytes = num_weights * out_elem_bytes;

    // Activation is TILE layout [1,1,1,H] -> Kt == k_tiles tiles (one tile-row).
    const uint32_t input_page_size = static_cast<uint32_t>(input_buffer->page_size());
    const uint32_t input_num_pages = static_cast<uint32_t>(input_buffer->num_pages());

    // Output is TILE [num_active, 1, I] bf16 (the SwiGLU activation): each active core
    // writes its 2 output tiles (its 64-column I slice) per expert. i_tiles == I/32 is
    // both the output page stride per expert row and the idle-core guard.

    const uint32_t out_tile_bytes = static_cast<uint32_t>(out_buffer->page_size());

    // SwiGLU clamp limit, passed to the compute kernel as a bit-cast float (the kernel
    // derives -limit internally).
    const uint32_t limit_bits = std::bit_cast<uint32_t>(operation_attributes.swiglu_limit);

    const uint32_t routing_cb_bytes = std::max<uint32_t>(align_up_32(routing_page_bytes), 32u);
    const uint32_t bcast_cb_bytes = std::max<uint32_t>(align_up_32(bcast_page_bytes), 32u);
    const uint32_t input_cb_bytes = input_num_pages * input_page_size;
    // Double-buffer the matmul output so compute can run ahead of the writer. Each core
    // produces kOutTilesPerCore SwiGLU output tiles (its 64-column I slice) per expert.
    constexpr uint32_t kOutTilesPerCore = 2;
    const uint32_t out_cb_bytes = 2u * kOutTilesPerCore * out_tile_bytes;
    // Matmul staging buffer (fp32 for full precision before the SwiGLU SFPU pass):
    // 2*kOutTilesPerCore tiles per expert (gate 0,1 | up 2,3), single-buffered.
    const uint32_t mm_tile_bytes = TILE_DIM * TILE_DIM * 4u;
    const uint32_t mm_cb_bytes = 2u * kOutTilesPerCore * mm_tile_bytes;

    // Core sets: full grid, the two senders {0,0} (expert ids) and {1,0} (activations),
    // and the 62 receivers.
    const CoreCoord sender{0, 0};
    const CoreCoord input_sender{1, 0};
    const CoreRange all_range({0, 0}, {GRID_X - 1, GRID_Y - 1});
    const CoreRangeSet all_cores{all_range};
    const CoreRangeSet sender_set{CoreRange{sender, sender}};
    const CoreRangeSet input_sender_set{CoreRange{input_sender, input_sender}};
    // Receivers = full grid minus {0,0} and {1,0}: row 0 (x=2..7) plus rows 1..7 (all x).
    const CoreRangeSet receiver_cores{std::vector<CoreRange>{
        CoreRange{{2, 0}, {GRID_X - 1, 0}},
        CoreRange{{0, 1}, {GRID_X - 1, GRID_Y - 1}},
    }};
    // Writers on the DM processor not used by each core's reader: {1,0}'s reader is
    // NoC 1, so its writer is NoC 0; everyone else's reader is NoC 0, writer NoC 1.
    const CoreRangeSet writer_noc1_cores{std::vector<CoreRange>{
        CoreRange{sender, sender},
        CoreRange{{2, 0}, {GRID_X - 1, 0}},
        CoreRange{{0, 1}, {GRID_X - 1, GRID_Y - 1}},
    }};
    const CoreRangeSet writer_noc0_cores{CoreRange{input_sender, input_sender}};

    ProgramDescriptor desc;

    // Two broadcast-ready semaphores on ALL cores: expert ids ({0,0}) and activations ({1,0}).
    constexpr uint32_t sem_id = 0;
    constexpr uint32_t sem_input_id = 1;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sem_id,
        .core_type = CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sem_input_id,
        .core_type = CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });

    // CBs are allocated identically on all cores so the broadcast CBs land at the same
    // L1 address everywhere (required for the multicast writes to be valid).
    constexpr uint32_t cb_routing = CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = routing_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_routing,
            .data_format = routing_df,
            .page_size = routing_cb_bytes,
        }}},
    });

    constexpr uint32_t cb_bcast = CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = bcast_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_bcast,
            .data_format = tt::DataFormat::UInt32,
            .page_size = bcast_cb_bytes,
        }}},
    });

    // Activation tiles (page = one tile) so the matmul can index them tile-by-tile.
    constexpr uint32_t cb_input = CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_input,
            .data_format = input_df,
            .page_size = input_page_size,
        }}},
    });

    // Per-core gate_up weight slice ([K, 128] = k_tiles x 4 tiles: gate 0,1 | up 2,3),
    // double-buffered.
    constexpr uint32_t cb_weights = CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = weights_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_weights,
            .data_format = gate_up_df,
            .page_size = weight_tile_bytes,
        }}},
    });

    // Per-core SwiGLU output (kOutTilesPerCore tiles per expert), double-buffered.
    constexpr uint32_t cb_out = CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_out,
            .data_format = out_df,
            .page_size = out_tile_bytes,
        }}},
    });

    // Per-core matmul staging buffer (fp32): the compute kernel packs the gate/up matmul
    // results here (gate 0,1 | up 2,3), then reloads them for the SwiGLU SFPU pass.
    constexpr uint32_t cb_mm = CBIndex::c_5;
    desc.cbs.push_back(CBDescriptor{
        .total_size = mm_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_mm,
            .data_format = tt::DataFormat::Float32,
            .page_size = mm_tile_bytes,
        }}},
    });

    // Multicast rectangle (NoC coords) covering the whole grid. Non-loopback
    // multicast excludes the sender, so num_dests = total cores - 1.
    const auto corner_a = device->worker_core_from_logical_core(CoreCoord{0, 0});
    const auto corner_b = device->worker_core_from_logical_core(CoreCoord{GRID_X - 1, GRID_Y - 1});
    const uint32_t mcast_start_x = std::min<uint32_t>(corner_a.x, corner_b.x);
    const uint32_t mcast_start_y = std::min<uint32_t>(corner_a.y, corner_b.y);
    const uint32_t mcast_end_x = std::max<uint32_t>(corner_a.x, corner_b.x);
    const uint32_t mcast_end_y = std::max<uint32_t>(corner_a.y, corner_b.y);
    const uint32_t num_dests = GRID_X * GRID_Y - 1;

    // Each core's first SwiGLU output tile is idx * kTilesPerCore (it owns the 2 I-dim
    // output tiles [idx*2, idx*2 + 1]), idx = y*GRID_X + x. The fetch derives the DRAM
    // shard id from this as col_start_tile / kTilesPerCore == idx.
    auto col_start_tile_for = [](const CoreCoord& c) -> uint32_t { return (c.y * GRID_X + c.x) * kTilesPerCore; };
    // Base address of every expert's gate_up weight, in expert-id order. All
    // experts are passed so the fetch can index by routing-selected hit id.
    std::vector<uint32_t> gate_up_addrs;
    gate_up_addrs.reserve(num_weights);
    for (const auto& w : tensor_args.gate_up_weights) {
        gate_up_addrs.push_back(static_cast<uint32_t>(w.buffer()->address()));
    }
    auto append_addrs = [&](KernelDescriptor::CoreRuntimeArgs& args) {
        for (uint32_t a : gate_up_addrs) {
            args.push_back(a);
        }
    };

    // ---- Expert-id sender kernel on {0,0} (NoC 0). ----
    std::vector<uint32_t> sender_ct_args = {
        num_weights,
        num_active,
        sentinel,
        cb_routing,
        cb_bcast,
        routing_page_bytes,
        bcast_page_bytes,
        sem_id,
        cb_weights,
        k_tiles,
        i_tiles,
        weight_tile_bytes,
        sem_input_id,
        cb_input,
    };
    TensorAccessorArgs(*routing_buffer).append_to(sender_ct_args);
    TensorAccessorArgs(*gate_up0_buffer).append_to(sender_ct_args);

    KernelDescriptor sender_desc;
    sender_desc.kernel_source = std::string(kKernelDir) + "/dataflow/compute_expert_ids.cpp";
    sender_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_desc.core_ranges = sender_set;
    sender_desc.compile_time_args = sender_ct_args;
    sender_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_0,
    };
    {
        KernelDescriptor::CoreRuntimeArgs args{
            routing_buffer->address(),
            mcast_start_x,
            mcast_start_y,
            mcast_end_x,
            mcast_end_y,
            num_dests,
            col_start_tile_for(sender),
        };
        append_addrs(args);
        sender_desc.runtime_args.emplace_back(sender, std::move(args));
    }
    desc.kernels.push_back(std::move(sender_desc));

    // ---- Input-broadcaster kernel on {1,0} (NoC 1). ----
    std::vector<uint32_t> input_ct_args = {
        cb_input,
        input_page_size,
        input_num_pages,
        sem_input_id,
        sem_id,
        num_active,
        cb_weights,
        k_tiles,
        i_tiles,
        weight_tile_bytes,
        cb_bcast,
    };
    TensorAccessorArgs(*input_buffer).append_to(input_ct_args);
    TensorAccessorArgs(*gate_up0_buffer).append_to(input_ct_args);

    KernelDescriptor input_sender_desc;
    input_sender_desc.kernel_source = std::string(kKernelDir) + "/dataflow/broadcast_input.cpp";
    input_sender_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    input_sender_desc.core_ranges = input_sender_set;
    input_sender_desc.compile_time_args = input_ct_args;
    input_sender_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_1,
    };
    // NoC 1 multicasts traverse from high to low coordinates, so swap start/end.
    {
        KernelDescriptor::CoreRuntimeArgs args{
            input_buffer->address(),
            mcast_end_x,
            mcast_end_y,
            mcast_start_x,
            mcast_start_y,
            num_dests,
            col_start_tile_for(input_sender),
        };
        append_addrs(args);
        input_sender_desc.runtime_args.emplace_back(input_sender, std::move(args));
    }
    desc.kernels.push_back(std::move(input_sender_desc));

    // ---- Receiver reader kernel on the other 62 cores (NoC 0). ----
    std::vector<uint32_t> receiver_ct_args = {
        sem_id,
        sem_input_id,
        num_active,
        cb_input,
        cb_weights,
        k_tiles,
        i_tiles,
        weight_tile_bytes,
        cb_bcast,
    };
    TensorAccessorArgs(*gate_up0_buffer).append_to(receiver_ct_args);

    KernelDescriptor receiver_desc;
    receiver_desc.kernel_source = std::string(kKernelDir) + "/dataflow/wait_expert_ids.cpp";
    receiver_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    receiver_desc.core_ranges = receiver_cores;
    receiver_desc.compile_time_args = receiver_ct_args;
    receiver_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_0,
    };
    for (const auto& cr : receiver_cores.ranges()) {
        for (const auto& core : cr) {
            KernelDescriptor::CoreRuntimeArgs args{col_start_tile_for(core)};
            append_addrs(args);
            receiver_desc.runtime_args.emplace_back(core, std::move(args));
        }
    }
    desc.kernels.push_back(std::move(receiver_desc));

    // ---- Compute (gate_up matmul) kernel on all 64 cores. ----
    std::vector<uint32_t> compute_ct_args = {
        num_active,
        k_tiles,
        i_tiles,
        cb_input,
        cb_weights,
        cb_mm,
        cb_out,
        limit_bits,
    };
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = std::string(kKernelDir) + "/compute/matmul_gate_up.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = compute_ct_args;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
    };
    for (uint32_t y = 0; y < GRID_Y; ++y) {
        for (uint32_t x = 0; x < GRID_X; ++x) {
            const CoreCoord core{x, y};
            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{col_start_tile_for(core)});
        }
    }
    desc.kernels.push_back(std::move(compute_desc));

    // ---- Writer kernel on all 64 cores (two processor groups). ----
    std::vector<uint32_t> writer_ct_args = {
        num_active,
        i_tiles,  // I/32: output page stride per expert row and the out-of-range guard
        cb_out,
        out_tile_bytes,
    };
    TensorAccessorArgs(*out_buffer).append_to(writer_ct_args);

    auto make_writer = [&](const CoreRangeSet& cores, DataMovementProcessor proc, NOC noc) {
        KernelDescriptor writer_desc;
        writer_desc.kernel_source = std::string(kKernelDir) + "/dataflow/write_gate_up.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = cores;
        writer_desc.compile_time_args = writer_ct_args;
        writer_desc.config = DataMovementConfigDescriptor{.processor = proc, .noc = noc};
        for (const auto& cr : cores.ranges()) {
            for (const auto& core : cr) {
                writer_desc.runtime_args.emplace_back(
                    core, KernelDescriptor::CoreRuntimeArgs{out_buffer->address(), col_start_tile_for(core)});
            }
        }
        desc.kernels.push_back(std::move(writer_desc));
    };
    make_writer(writer_noc1_cores, DataMovementProcessor::RISCV_1, NOC::NOC_1);
    make_writer(writer_noc0_cores, DataMovementProcessor::RISCV_0, NOC::NOC_0);

    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts
