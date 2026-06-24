// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::data_movement {

Fold::program_factory_t Fold::select_program_factory(
    const operation_attributes_t& op_attr, const tensor_args_t& /*tensors*/) {
    // `MultiCore` is RM-only zero-NOC; everything else uses TensorAccessor-based MultiCoreDRAMFold.
    if (op_attr.is_height_sharded_rm_fast_path) {
        return MultiCore{};
    }
    return MultiCoreDRAMFold{};
}

void validate_fold(
    const std::vector<Tensor>& input_tensors,
    bool is_height_sharded_rm_fast_path,
    uint32_t stride_h,
    uint32_t stride_w) {
    const Tensor& input_tensor = input_tensors.at(0);

    const auto& input_shape = input_tensor.padded_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Fold: Expect input tensor to be stored on device.");
    TT_FATAL(input_tensor.buffer() != nullptr, "Fold: Expect input tensor to be allocated on a device buffer.");

    // Mathematical fold constraints — apply to every path.
    TT_FATAL(input_shape[1] % stride_h == 0, "Fold: Input height must be divisible by stride_h.");
    TT_FATAL(input_shape[2] % stride_w == 0, "Fold: Input width must be divisible by stride_w.");

    // Fast-path also requires the stride_h x stride_w neighbourhood to live on one core (RM, sticks).
    if (is_height_sharded_rm_fast_path) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Fold (RM fast path): expects HEIGHT_SHARDED memory layout.");
        TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "Fold (RM fast path): requires ROW_MAJOR layout.");
        auto shard_shape = input_tensor.shard_spec().value().shape;
        TT_FATAL(
            shard_shape[0] % (input_shape[2] * stride_h) == 0,
            "Fold (RM fast path): shard height must be divisible by input_width * stride_h.");
    }
}

void Fold::validate_on_program_cache_miss(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    validate_fold({tensors.input_tensor}, op_attr.is_height_sharded_rm_fast_path, op_attr.stride_h, op_attr.stride_w);
}

void Fold::validate_on_program_cache_hit(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    validate_fold({tensors.input_tensor}, op_attr.is_height_sharded_rm_fast_path, op_attr.stride_h, op_attr.stride_w);
}

Fold::spec_return_value_t Fold::compute_output_specs(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    auto input_tensor = tensors.input_tensor;
    const ttnn::Shape& input_shape = input_tensor.logical_shape();
    auto input_dtype = input_tensor.dtype();

    tt::tt_metal::DataType output_dtype =
        (input_dtype == tt::tt_metal::DataType::FLOAT32 ||
         input_dtype == tt::tt_metal::DataType::UINT16)
            ? input_dtype
            : tt::tt_metal::DataType::BFLOAT16;

    // Folded 4D: (N, H/sh, W/sw, C*sh*sw).
    const ttnn::Shape folded_4d_shape(
        {input_shape[0],
         input_shape[1] / op_attr.stride_h,
         input_shape[2] / op_attr.stride_w,
         input_shape[3] * op_attr.stride_h * op_attr.stride_w});

    // Legacy collapsed: (1, 1, N*H/sh*W/sw, C*sh*sw).
    const ttnn::Shape collapsed_shape(
        {1,
         1,
         input_shape[0] * input_shape[1] * input_shape[2] / (op_attr.stride_h * op_attr.stride_w),
         input_shape[3] * op_attr.stride_h * op_attr.stride_w});

    if (op_attr.is_height_sharded_rm_fast_path) {
        // HEIGHT_SHARDED + RM fast path: keep collapsed sharded output (legacy).
        auto shard_spec = input_tensor.shard_spec().value();
        shard_spec.shape[0] /= op_attr.stride_h * op_attr.stride_w;
        shard_spec.shape[1] *= op_attr.stride_h * op_attr.stride_w;
        auto mem_config = MemoryConfig(
            input_tensor.memory_config().memory_layout(), input_tensor.memory_config().buffer_type(), shard_spec);

        return {TensorSpec(
            collapsed_shape,
            tt::tt_metal::TensorLayout(
                output_dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), mem_config))};
    }

    // For TILE inputs we preserve the input 4D shape; composite in fold.cpp reshapes to folded_4d.
    const bool input_is_tile = input_tensor.layout() == Layout::TILE;
    const ttnn::Shape preserved_4d_shape({input_shape[0], input_shape[1], input_shape[2], input_shape[3]});

    if (input_tensor.is_sharded()) {
        // HEIGHT+TILE, W/B-sharded (any layout) → L1 interleaved (fold has no natural W/B output).
        const ttnn::Shape output_logical_shape = input_is_tile ? preserved_4d_shape : folded_4d_shape;
        auto out_mc = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1);
        return {TensorSpec(
            output_logical_shape,
            tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), out_mc))};
    }

    // Interleaved (DRAM or L1): TILE → preserved_4d; DRAM RM → folded_4d; L1 RM → collapsed (legacy).
    ttnn::Shape output_logical_shape;
    if (input_is_tile) {
        output_logical_shape = preserved_4d_shape;
    } else if (input_tensor.memory_config().is_dram()) {
        output_logical_shape = folded_4d_shape;
    } else {
        output_logical_shape = collapsed_shape;
    }
    return {TensorSpec(
        output_logical_shape,
        tt::tt_metal::TensorLayout(
            output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), input_tensor.memory_config()))};
}

Fold::tensor_return_value_t Fold::create_output_tensors(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    return create_device_tensor(compute_output_specs(op_attr, tensors), tensors.input_tensor.device());
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::Fold::tensor_return_value_t fold(
    const ttnn::Tensor& input_tensor, uint32_t stride_h, uint32_t stride_w) {
    using OperationType = ttnn::operations::data_movement::Fold;
    // Fast path is HEIGHT_SHARDED + ROW_MAJOR; every other combo flows through MultiCoreDRAMFold.
    const bool is_height_sharded_rm_fast_path =
        input_tensor.is_sharded() &&
        input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED &&
        input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR;
    auto operation_attributes = OperationType::operation_attributes_t{
        .stride_h = stride_h, .stride_w = stride_w, .is_height_sharded_rm_fast_path = is_height_sharded_rm_fast_path};
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
