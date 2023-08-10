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

#include "xla/service/gpu/gpu_fused_mha_runner.h"

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/half_type.hpp>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/xetla/sdp/sdp.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/sycl/sycl_stream.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

using bfloat16 = sycl::ext::oneapi::bfloat16;
using half = sycl::half;

namespace {
using se::DeviceMemory;
using se::DeviceMemoryBase;
using se::dnn::DataType;
using se::dnn::MatmulTensorDescriptor;
using se::dnn::TensorDescriptor;

template <typename ElementType, typename BiasType, typename OutputType>
Status RunFusedMHAScaleBiasSoftmax(GpufMHAParams params, se::Stream* stream,
                                   DeviceMemory<ElementType> lhs_bmm1_buffer,
                                   DeviceMemory<ElementType> rhs_bmm1_buffer,
                                   DeviceMemory<ElementType> rhs_bmm2_buffer,
                                   DeviceMemory<OutputType> output_buffer,
                                   DeviceMemory<BiasType> bias_buffer,
                                   DeviceMemoryBase scratch_memory) {
  sycl::queue* dpcpp_stream = se::gpu::AsGpuStreamValue(stream);
  std::optional<double> dropout_rate;
  if (params.config->dropout_rate.has_value()) {
    dropout_rate = *params.config->dropout_rate;
    VLOG(1) << "dropout_rate: " << *dropout_rate;
  }

  float scale = 1.0;
  if (params.config->fmha_scale.has_value()) {
    scale = static_cast<float>(*params.config->fmha_scale);
    VLOG(1) << "scale: " << scale;
  }

  std::optional<int64_t> seed;
  if (params.config->seed.has_value()) {
    seed = *params.config->seed;
    VLOG(1) << "seed: " << *seed;
  }

  auto lhs_bmm1_desc = params.config->lhs_bmm1;
  auto rhs_bmm1_desc = params.config->rhs_bmm1;
  auto rhs_bmm2_desc = params.config->rhs_bmm2;
  auto output_desc = params.config->output;
  VLOG(1) << "lhs_bmm1_desc: \n" << lhs_bmm1_desc.ToString();
  VLOG(1) << "rhs_bmm1_desc: \n" << rhs_bmm1_desc.ToString();
  VLOG(1) << "rhs_bmm2_desc: \n" << rhs_bmm2_desc.ToString();
  VLOG(1) << "output_desc: \n" << output_desc.ToString();
  if (params.config->bias.has_value()) {
    auto bias_desc = *params.config->bias;
    VLOG(1) << "bias_desc: \n" << bias_desc.ToString();
  }

  auto lhs_bmm1_dims =
      lhs_bmm1_desc.GetCudnnCompatibleDimensions(/*is_lhs*/ true);
  auto rhs_bmm1_dims =
      rhs_bmm1_desc.GetCudnnCompatibleDimensions(/*is_lhs*/ false);
  auto rhs_bmm2_dims =
      rhs_bmm2_desc.GetCudnnCompatibleDimensions(/*is_lhs*/ false);
  VLOG(1) << "lhs_bmm1_dims: " << absl::StrJoin(lhs_bmm1_dims, ",");
  VLOG(1) << "rhs_bmm1_dims: " << absl::StrJoin(rhs_bmm1_dims, ",");
  VLOG(1) << "rhs_bmm2_dims: " << absl::StrJoin(rhs_bmm2_dims, ",");
  auto lhs_bmm1_strides =
      lhs_bmm1_desc.GetCudnnCompatibleStrides(/*is_lhs*/ true);
  auto rhs_bmm1_strides =
      rhs_bmm1_desc.GetCudnnCompatibleStrides(/*is_lhs*/ false);
  auto rhs_bmm2_strides =
      rhs_bmm2_desc.GetCudnnCompatibleStrides(/*is_lhs*/ false);
  VLOG(1) << "lhs_bmm1_strides: " << absl::StrJoin(lhs_bmm1_strides, ",");
  VLOG(1) << "rhs_bmm1_strides: " << absl::StrJoin(rhs_bmm1_strides, ",");
  VLOG(1) << "rhs_bmm2_strides: " << absl::StrJoin(rhs_bmm2_strides, ",");

  int rank = lhs_bmm1_strides.size();
  CHECK(lhs_bmm1_strides[rank - 1] == 1);
  CHECK(rhs_bmm1_strides[rank - 2] == 1);
  CHECK(rhs_bmm2_strides[rank - 1] == 1);
  // [B,N,F,H] * [B,N,T,H] * [B,N,T,H]
  // Assume rhs_bmm1 is transposed
  int B = (rank == 4) ? lhs_bmm1_dims[rank - 4] : 1;
  int N = lhs_bmm1_dims[rank - 3];
  int F = lhs_bmm1_dims[rank - 2];
  int H = lhs_bmm1_dims[rank - 1];
  int T = rhs_bmm1_dims[rank - 1];

  auto lhs_bmm1_ptr = reinterpret_cast<void*>(lhs_bmm1_buffer.opaque());
  auto rhs_bmm1_ptr = reinterpret_cast<void*>(rhs_bmm1_buffer.opaque());
  auto rhs_bmm2_ptr = reinterpret_cast<void*>(rhs_bmm2_buffer.opaque());
  auto output_ptr = reinterpret_cast<void*>(output_buffer.opaque());
  auto bias_ptr = reinterpret_cast<void*>(bias_buffer.opaque());

  // Recalculate scale since scale attr has accuracy issue.
  if ((scale - 1.0f) > 1e-6) scale = 1.0f / sqrt(H);
  if (std::is_same_v<ElementType, bfloat16>) {
    if (bias_ptr)
      ::gpu::xetla::fmha_forward_bf16_bias(
          *dpcpp_stream, lhs_bmm1_ptr, rhs_bmm1_ptr, rhs_bmm2_ptr, bias_ptr,
          nullptr, 1.0f, output_ptr, B, N, H, F, T, scale);
    else
      ::gpu::xetla::fmha_forward_bf16(*dpcpp_stream, lhs_bmm1_ptr, rhs_bmm1_ptr,
                                      rhs_bmm2_ptr, bias_ptr, nullptr, 1.0f,
                                      output_ptr, B, N, H, F, T, scale);
  } else if (std::is_same_v<ElementType, half>) {
    if (bias_ptr)
      ::gpu::xetla::fmha_forward_fp16_bias(
          *dpcpp_stream, lhs_bmm1_ptr, rhs_bmm1_ptr, rhs_bmm2_ptr, bias_ptr,
          nullptr, 1.0f, output_ptr, B, N, H, F, T, scale);
    else
      ::gpu::xetla::fmha_forward_fp16(*dpcpp_stream, lhs_bmm1_ptr, rhs_bmm1_ptr,
                                      rhs_bmm2_ptr, bias_ptr, nullptr, 1.0f,
                                      output_ptr, B, N, H, F, T, scale);
  } else {
    return InternalError("Invalid MHA datatype");
  }
  return OkStatus();
}

template <typename ElementType, typename BiasType, typename OutputType>
Status RunGpuFMHAImpl(const GpufMHAParams& params, se::Stream* stream,
                      se::DeviceMemoryBase scratch_memory) {
  auto lhs_bmm1_buffer = se::DeviceMemory<ElementType>(params.lhs_bmm1_buffer);
  auto rhs_bmm1_buffer = se::DeviceMemory<ElementType>(params.rhs_bmm1_buffer);
  auto rhs_bmm2_buffer = se::DeviceMemory<ElementType>(params.rhs_bmm2_buffer);
  auto output_buffer = se::DeviceMemory<OutputType>(params.output_buffer);
  auto bias_buffer = params.bias_buffer.has_value()
                         ? se::DeviceMemory<BiasType>(*params.bias_buffer)
                         : se::DeviceMemory<BiasType>();

  se::dnn::AlgorithmDesc algorithm = params.config->algorithm;
  Status run_status = OkStatus();
  switch (params.config->kind) {
    case CudnnfMHAKind::kSoftmax:
    case CudnnfMHAKind::kScaleSoftmax:
    case CudnnfMHAKind::kScaleBiasSoftmax:
    case CudnnfMHAKind::kScaleBiasSoftmaxDropout:
      run_status =
          RunFusedMHAScaleBiasSoftmax<ElementType, BiasType, OutputType>(
              params, stream, lhs_bmm1_buffer, rhs_bmm1_buffer, rhs_bmm2_buffer,
              output_buffer, bias_buffer, scratch_memory);
      break;
    default:
      return InternalError("Invalid cuDNN fMHA kind: %s",
                           CudnnfMHAKindToString(params.config->kind));
  }

  if (run_status != OkStatus()) {
    return run_status;
  }

  if (!stream->ok()) {
    return InternalError("Unable to launch FMHA with type %s",
                         CudnnfMHAKindToString(params.config->kind));
  }

  return OkStatus();
}
}  // namespace

/*static*/ StatusOr<GpufMHAConfig> GpufMHAConfig::For(
    const GpufMHADescriptor& desc) {
  // Get shapes from desc.
  const Shape& lhs_bmm1_shape = desc.lhs_bmm1_shape;
  const Shape& rhs_bmm1_shape = desc.rhs_bmm1_shape;
  const Shape& rhs_bmm2_shape = desc.rhs_bmm2_shape;
  const Shape& intermediate_lhs_bmm2_shape = desc.intermediate_lhs_bmm2_shape;
  const Shape& output_shape = desc.output_shape;

  // Get DNN dtype from primtive types
  TF_ASSIGN_OR_RETURN(
      DataType lhs_bmm1_type,
      GetDNNDataTypeFromPrimitiveType(lhs_bmm1_shape.element_type()));
  TF_ASSIGN_OR_RETURN(
      DataType rhs_bmm1_type,
      GetDNNDataTypeFromPrimitiveType(rhs_bmm1_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType rhs_bmm2_type,
      GetDNNDataTypeFromPrimitiveType(rhs_bmm2_shape.element_type()));
  TF_ASSIGN_OR_RETURN(DataType lhs_bmm2_type,
                      GetDNNDataTypeFromPrimitiveType(
                          intermediate_lhs_bmm2_shape.element_type()));
  TF_ASSIGN_OR_RETURN(DataType output_type, GetDNNDataTypeFromPrimitiveType(
                                                output_shape.element_type()));
  GpufMHAConfig config;
  config.input_type = lhs_bmm1_shape.element_type();
  config.output_type = output_shape.element_type();

  // Get MatmulTensorDescriptors for BMM1
  config.lhs_bmm1 =
      MatmulTensorDescriptor::For(lhs_bmm1_type, lhs_bmm1_shape.dimensions(),
                                  desc.lhs_bmm1_shape.layout().minor_to_major(),
                                  desc.bmm1_dnums.lhs_batch_dimensions(),
                                  desc.bmm1_dnums.lhs_contracting_dimensions());
  config.rhs_bmm1 =
      MatmulTensorDescriptor::For(rhs_bmm1_type, rhs_bmm1_shape.dimensions(),
                                  desc.rhs_bmm1_shape.layout().minor_to_major(),
                                  desc.bmm1_dnums.rhs_batch_dimensions(),
                                  desc.bmm1_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for BMM2
  config.rhs_bmm2 =
      MatmulTensorDescriptor::For(rhs_bmm2_type, rhs_bmm2_shape.dimensions(),
                                  desc.rhs_bmm2_shape.layout().minor_to_major(),
                                  desc.bmm2_dnums.rhs_batch_dimensions(),
                                  desc.bmm2_dnums.rhs_contracting_dimensions());

  config.intermediate_lhs_bmm2 = MatmulTensorDescriptor::For(
      lhs_bmm2_type, intermediate_lhs_bmm2_shape.dimensions(),
      desc.intermediate_lhs_bmm2_shape.layout().minor_to_major(),
      desc.bmm2_dnums.lhs_batch_dimensions(),
      desc.bmm2_dnums.lhs_contracting_dimensions());

  config.output = TensorDescriptor::For(output_type, output_shape.dimensions(),
                                        output_shape.layout().minor_to_major());

  config.kind = desc.kind;
  const CudnnfMHABackendConfig& backend_config = desc.backend_config;
  config.algorithm = se::dnn::AlgorithmDesc(backend_config.algorithm());

  auto check_and_assign_mask = [&]() -> Status {
    if (desc.mask_shape) {
      const Shape& mask_shape = *desc.mask_shape;

      TF_ASSIGN_OR_RETURN(DataType mask_type, GetDNNDataTypeFromPrimitiveType(
                                                  mask_shape.element_type()));
      config.mask = TensorDescriptor::For(mask_type, mask_shape.dimensions(),
                                          mask_shape.layout().minor_to_major());
      return OkStatus();
    } else {
      return InternalError(
          "GpufMHADescriptor should have non-nul mask shape but found null "
          "mask shape");
    }
  };

  auto check_and_assign_bias = [&]() -> Status {
    if (desc.bias_shape) {
      const Shape& bias_shape = *desc.bias_shape;

      TF_ASSIGN_OR_RETURN(DataType bias_type, GetDNNDataTypeFromPrimitiveType(
                                                  bias_shape.element_type()));

      config.bias = TensorDescriptor::For(bias_type, bias_shape.dimensions(),
                                          bias_shape.layout().minor_to_major());
      return OkStatus();
    } else {
      return InternalError(
          "GpufMHADescriptor should have non-nul bias shape but found null "
          "bias shape");
    }
  };

  auto assign_scale = [&]() {
    config.fmha_scale.emplace();
    double& fmha_scale = *config.fmha_scale;
    fmha_scale = backend_config.fmha_scale();
  };

  auto assign_dropout_rate = [&]() {
    config.dropout_rate.emplace();
    double& dropout_rate = *config.dropout_rate;
    dropout_rate = backend_config.dropout_rate();
  };

  auto assign_seed = [&]() {
    config.seed.emplace();
    int64_t& seed = *config.seed;
    seed = backend_config.seed();
  };

  switch (config.kind) {
    case CudnnfMHAKind::kScaleBiasMaskSoftmax:
      TF_RETURN_IF_ERROR(check_and_assign_mask());
      TF_RETURN_IF_ERROR(check_and_assign_bias());
      assign_scale();
      break;
    case CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout:
      TF_RETURN_IF_ERROR(check_and_assign_mask());
      TF_RETURN_IF_ERROR(check_and_assign_bias());
      assign_scale();
      assign_dropout_rate();
      assign_seed();
      break;
    case CudnnfMHAKind::kScaleMaskSoftmax:
      TF_RETURN_IF_ERROR(check_and_assign_mask());
      assign_scale();
      break;
    case CudnnfMHAKind::kScaleMaskSoftmaxDropout:
      TF_RETURN_IF_ERROR(check_and_assign_mask());
      assign_scale();
      assign_dropout_rate();
      assign_seed();
      break;
    case CudnnfMHAKind::kBmmBmm:
    case CudnnfMHAKind::kSoftmax:
      break;
    case CudnnfMHAKind::kSoftmaxDropout:
      assign_dropout_rate();
      assign_seed();
      break;
    case CudnnfMHAKind::kScaleBiasSoftmaxDropout:
      TF_RETURN_IF_ERROR(check_and_assign_bias());
      assign_scale();
      assign_dropout_rate();
      assign_seed();
      break;
    case CudnnfMHAKind::kScaleBiasSoftmax:
      TF_RETURN_IF_ERROR(check_and_assign_bias());
      assign_scale();
      break;
    case CudnnfMHAKind::kScaleSoftmax:
      assign_scale();
      break;
    default:
      return InternalError("Unknown fmha kind");
  }
  return config;
}

/*static*/ StatusOr<GpufMHAParams> GpufMHAParams::For(
    const GpufMHAConfig& config, se::DeviceMemoryBase lhs_bmm1_buffer,
    se::DeviceMemoryBase rhs_bmm1_buffer, se::DeviceMemoryBase rhs_bmm2_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase mask_buffer,
    se::DeviceMemoryBase bias_buffer) {
  GpufMHAParams params;
  params.config = &config;
  params.lhs_bmm1_buffer = lhs_bmm1_buffer;
  params.rhs_bmm1_buffer = rhs_bmm1_buffer;
  params.rhs_bmm2_buffer = rhs_bmm2_buffer;
  params.output_buffer = output_buffer;

  auto assign_mask_buffer = [&]() {
    params.mask_buffer.emplace();
    se::DeviceMemoryBase& mask = *params.mask_buffer;
    mask = mask_buffer;
  };

  auto assign_bias_buffer = [&]() {
    params.bias_buffer.emplace();
    se::DeviceMemoryBase& bias = *params.bias_buffer;
    bias = bias_buffer;
  };

  switch (config.kind) {
    case CudnnfMHAKind::kBmmBmm:
    case CudnnfMHAKind::kSoftmaxDropout:
    case CudnnfMHAKind::kSoftmax:
      break;
    case CudnnfMHAKind::kScaleMaskSoftmax:
    case CudnnfMHAKind::kScaleMaskSoftmaxDropout:
      TF_RET_CHECK(!mask_buffer.is_null());
      assign_mask_buffer();
      break;
    case CudnnfMHAKind::kScaleBiasMaskSoftmax:
    case CudnnfMHAKind::kScaleBiasMaskSoftmaxDropout:
      TF_RET_CHECK(!mask_buffer.is_null());
      TF_RET_CHECK(!bias_buffer.is_null());
      assign_mask_buffer();
      assign_bias_buffer();
      break;
    case CudnnfMHAKind::kScaleBiasSoftmax:
    case CudnnfMHAKind::kScaleBiasSoftmaxDropout:
      TF_RET_CHECK(!bias_buffer.is_null());
      assign_bias_buffer();
  }
  return params;
}

Status RunGpuFMHA(const GpufMHAConfig& fmha_config,
                  se::DeviceMemoryBase lhs_bmm1_buffer,
                  se::DeviceMemoryBase rhs_bmm1_buffer,
                  se::DeviceMemoryBase rhs_bmm2_buffer,
                  se::DeviceMemoryBase output_buffer,
                  se::DeviceMemoryBase scratch_buffer,
                  se::DeviceMemoryBase mask_buffer,
                  se::DeviceMemoryBase bias_buffer, se::Stream* stream) {
  TF_ASSIGN_OR_RETURN(
      GpufMHAParams params,
      GpufMHAParams::For(fmha_config, lhs_bmm1_buffer, rhs_bmm1_buffer,
                         rhs_bmm2_buffer, output_buffer, mask_buffer,
                         bias_buffer));
  PrimitiveType input_primitive_type = fmha_config.input_type;
  switch (input_primitive_type) {
    case F16:
      return RunGpuFMHAImpl<half, half, half>(params, stream, scratch_buffer);
    case BF16:
      return RunGpuFMHAImpl<bfloat16, bfloat16, bfloat16>(params, stream,
                                                          scratch_buffer);
    default:
      return Unimplemented("Unimplemented fused MHA");
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
