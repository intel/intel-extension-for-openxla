/* Copyright (c) 2023 Intel Corporation

Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/gpu_fused_qkv_runner.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/xetla/gemm/gemm.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/sycl/sycl_stream.h"
#include "xla/util.h"

// #include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/half_type.hpp>

namespace xla {
namespace gpu {

using se::DeviceMemory;
using se::DeviceMemoryBase;

/*static*/ StatusOr<GpufQKVConfig> GpufQKVConfig::For(
    const GpufQKVDescriptor& desc) {
  // Get shapes from desc.
  VLOG(2) << "Set GpufQKVConfig from GpufQKVDescriptor!!!\n";
  PrimitiveType element_type = desc.in_shape.element_type();
  Shape in_shape =
      desc.in_shape.rank() == 3
          ? desc.in_shape
          : ShapeUtil::MakeShape(element_type,
                                 {1, /*m = */ desc.in_shape.dimensions(0),
                                  /*k = */ desc.in_shape.dimensions(1)});

  Shape wei_shape = desc.wei_shape;
  TF_RET_CHECK(wei_shape.rank() == 3);
  // 3 output have same shape, only select one of them to comp
  Shape output_shape =
      desc.out1_shape.rank() == 3
          ? desc.out1_shape
          : ShapeUtil::MakeShape(element_type,
                                 {1, /*m = */ desc.out1_shape.dimensions(0),
                                  /*n = */ desc.out1_shape.dimensions(1)});
  VLOG(2) << "GpufQKVConfig, Input Shape: " << in_shape.ToString(true);
  VLOG(2) << "GpufQKVConfig, Wei Shape: " << wei_shape.ToString(true);
  VLOG(2) << "GpufQKVConfig, Out1 Shape: " << output_shape.ToString(true);
  // TODO: add support for qkv_bias
  // auto check_and_assign_bias = [&]() -> Status {
  //   if (desc.bias_shape) {
  //     const Shape& bias_shape = *desc.bias_shape;

  //     TF_ASSIGN_OR_RETURN(DataType bias_type,
  //     GetDNNDataTypeFromPrimitiveType(
  //                                                 bias_shape.element_type()));

  //     config.bias = TensorDescriptor::For(bias_type, bias_shape.dimensions(),
  //                                         bias_shape.layout().minor_to_major());
  //     return OkStatus();
  //   } else {
  //     return InternalError(
  //         "GpufQKVDescriptor should have non-nul bias shape but found null "
  //         "bias shape");
  //   }
  // };
  TF_ASSIGN_OR_RETURN(MatrixLayout in_layout, MatrixLayout::For(in_shape));

  TF_ASSIGN_OR_RETURN(MatrixLayout wei_layout, MatrixLayout::For(wei_shape));

  TF_ASSIGN_OR_RETURN(MatrixLayout out1_layout,
                      MatrixLayout::For(output_shape));

  TF_ASSIGN_OR_RETURN(MatrixLayout out2_layout,
                      MatrixLayout::For(output_shape));

  TF_ASSIGN_OR_RETURN(MatrixLayout out3_layout,
                      MatrixLayout::For(output_shape));

  TF_RET_CHECK((in_layout.batch_size == out1_layout.batch_size) &&
               (in_layout.batch_size == 1));
  TF_RET_CHECK((wei_layout.batch_size == 3));

  switch (output_shape.element_type()) {
    case F16:
      break;
    default:
      return InternalError("Unexpected GEMM datatype: %s",
                           primitive_util::LowercasePrimitiveTypeName(
                               output_shape.element_type()));
  }

  return GpufQKVConfig{
      in_layout, wei_layout, out1_layout, out2_layout, out3_layout,
  };
}

MatrixDescriptor GetMatrixDescriptor(const MatrixLayout& layout,
                                     se::DeviceMemoryBase data) {
  bool transpose = layout.order == MatrixLayout::Order::kColumnMajor;
  return {
      data,
      transpose ? se::blas::Transpose::kTranspose
                : se::blas::Transpose::kNoTranspose,
      transpose ? layout.num_cols : layout.num_rows,
      transpose ? layout.num_rows : layout.num_cols,
      layout.batch_stride,
      layout.leading_dim_stride,
  };
}

template <typename ElementType, typename OutputType>
Status RunGpuFQKVImpl(se::Stream* stream, const MatrixDescriptor& in_buffer,
                      const MatrixDescriptor& wei_buffer,
                      const MatrixDescriptor& out1_buffer,
                      const MatrixDescriptor& out2_buffer,
                      const MatrixDescriptor& out3_buffer, const int64_t m,
                      const int64_t n, const int64_t k, bool is_b_row_major) {
  sycl::queue q = *se::gpu::AsGpuStreamValue(stream);
  auto policy = ::gpu::xetla::XetlaQKVGemmKernel<ElementType>()
                    .add_matrix_q_out(out1_buffer)
                    .add_matrix_k_out(out2_buffer)
                    .add_matrix_v_out(out3_buffer)
                    .add_matrix_a(in_buffer)
                    .add_matrix_b(wei_buffer)
                    .build();
  if (policy.fallback() == false) {
    policy.run(&q);
  }
  return OkStatus();
}

Status RunGpuFQKV(const GpufQKVConfig& config, se::DeviceMemoryBase in_buffer,
                  se::DeviceMemoryBase wei_buffer,
                  se::DeviceMemoryBase out1_buffer,
                  se::DeviceMemoryBase out2_buffer,
                  se::DeviceMemoryBase out3_buffer, se::Stream* stream) {
  VLOG(1) << "Executing a FusedQKVThunk";

  MatrixLayout in_layout = config.in_layout;    // [m, k]
  MatrixLayout wei_layout = config.wei_layout;  // [3, k, n]
  MatrixLayout out1_layout = config.out1_layout;
  MatrixLayout out2_layout = config.out2_layout;
  MatrixLayout out3_layout = config.out3_layout;

  const int64_t m = in_layout.num_rows;
  const int64_t k = in_layout.num_cols;

  const int64_t n = wei_layout.num_cols;  // last dim

  CHECK(wei_layout.batch_size == 3);
  CHECK(out1_layout.num_rows == m && out2_layout.num_rows == m &&
        out3_layout.num_rows == m);
  CHECK(out1_layout.num_cols == n && out2_layout.num_cols == n &&
        out3_layout.num_cols == n);

  if ((out1_layout.dtype == F16) && (in_layout.dtype != out1_layout.dtype ||
                                     wei_layout.dtype != out1_layout.dtype)) {
    return InternalError(
        "GEMM lhs type(%s) and rhs type(%s) must match output type(%s)",
        primitive_util::LowercasePrimitiveTypeName(in_layout.dtype),
        primitive_util::LowercasePrimitiveTypeName(wei_layout.dtype),
        primitive_util::LowercasePrimitiveTypeName(out1_layout.dtype));
  }
  bool is_b_row_major = (wei_layout.order == MatrixLayout::Order::kRowMajor);
  PrimitiveType input_primitive_type = in_layout.dtype;
  switch (input_primitive_type) {
    case F16:
      return RunGpuFQKVImpl<sycl::half, sycl::half>(
          stream, GetMatrixDescriptor(in_layout, in_buffer),
          GetMatrixDescriptor(wei_layout, wei_buffer),
          GetMatrixDescriptor(out1_layout, out1_buffer),
          GetMatrixDescriptor(out2_layout, out2_buffer),
          GetMatrixDescriptor(out2_layout, out3_buffer), m, n, k,
          is_b_row_major);
    default:
      return Unimplemented("Unimplemented fused QKV");
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
