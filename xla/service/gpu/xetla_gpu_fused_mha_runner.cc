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

#include "xla/service/gpu/xetla_gpu_fused_mha_runner.h"

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

template <typename ElementType, typename BiasType, typename OutputType>
absl::Status RunFusedMHA(GpufMHAParams params, se::Stream* stream,
                                   DeviceMemory<ElementType> lhs_bmm1_buffer,
                                   DeviceMemory<ElementType> rhs_bmm1_buffer,
                                   DeviceMemory<ElementType> rhs_bmm2_buffer,
                                   DeviceMemory<OutputType> output_buffer,
                                   DeviceMemoryBase mask_buffer,
                                   DeviceMemoryBase bias_buffer,
                                   DeviceMemoryBase scratch_memory,
                                   DeviceMemoryBase activation_output) {
  sycl::queue* dpcpp_stream = se::gpu::AsGpuStreamValue(stream);
  std::optional<double> dropout_rate;
  if (params.config->dropout_rate) {
    dropout_rate = *params.config->dropout_rate;
    VLOG(1) << "dropout_rate: " << *dropout_rate;
  }

  float scale = 1.0;
  if (params.config->fmha_scale) {
    scale = static_cast<float>(*params.config->fmha_scale);
    VLOG(1) << "scale: " << scale;
  }

  std::optional<int64_t> seed;
  if (params.config->seed) {
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
  if (params.config->bias) {
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
    return Internal("Invalid MHA datatype");
  }
  return absl::OkStatus();
}

template <typename ElementType, typename BiasType, typename OutputType>
absl::Status RunGpuFMHAImpl(const GpufMHAParams& params, se::Stream* stream,
                      se::DeviceMemoryBase scratch_memory) {
  auto lhs_bmm1_buffer = se::DeviceMemory<ElementType>(params.lhs_bmm1_buffer);
  auto rhs_bmm1_buffer = se::DeviceMemory<ElementType>(params.rhs_bmm1_buffer);
  auto rhs_bmm2_buffer = se::DeviceMemory<ElementType>(params.rhs_bmm2_buffer);
  auto output_buffer = se::DeviceMemory<OutputType>(params.output_buffer);
  auto activation_buffer =
      params.activation_buffer.has_value()
          ? se::DeviceMemory<OutputType>(*params.activation_buffer)
          : se::DeviceMemoryBase();
  auto mask_buffer = params.mask_buffer.has_value()
                         ? se::DeviceMemory<ElementType>(*params.mask_buffer)
                         : se::DeviceMemoryBase();
  auto bias_buffer = params.bias_buffer.has_value()
                         ? se::DeviceMemory<BiasType>(*params.bias_buffer)
                         : se::DeviceMemoryBase();

  se::dnn::AlgorithmDesc algorithm = params.config->algorithm;
  absl::Status run_status = absl::OkStatus();
  switch (params.config->kind) {
    case CudnnfMHAKind::kSoftmax:
    case CudnnfMHAKind::kScaleBiasSoftmax:
      run_status =
          RunFusedMHA<ElementType, BiasType, OutputType>(
              params, stream, lhs_bmm1_buffer, rhs_bmm1_buffer, rhs_bmm2_buffer,
              output_buffer, mask_buffer, bias_buffer, scratch_memory,
              activation_buffer);
      break;
    default:
      return Internal("Invalid cuDNN fMHA kind: %s",
                           CudnnfMHAKindToString(params.config->kind));
  }

  if (run_status != absl::OkStatus()) {
    return run_status;
  }

  if (!stream->ok()) {
    return Internal("Unable to launch FMHA with type %s",
                         CudnnfMHAKindToString(params.config->kind));
  }

  return OkStatus();
}
}  // namespace

absl::Status RunXetlaGpuFMHA(
    const GpufMHAConfig& fmha_config, se::DeviceMemoryBase lhs_bmm1_buffer,
    se::DeviceMemoryBase rhs_bmm1_buffer, se::DeviceMemoryBase rhs_bmm2_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase scratch_buffer,
    std::optional<se::DeviceMemoryBase> mask_buffer,
    std::optional<se::DeviceMemoryBase> bias_buffer,
    std::optional<se::DeviceMemoryBase> activation_buffer, se::Stream* stream) {
  TF_ASSIGN_OR_RETURN(
      GpufMHAParams params,
      GpufMHAParams::For(fmha_config, lhs_bmm1_buffer, rhs_bmm1_buffer,
                         rhs_bmm2_buffer, output_buffer, mask_buffer,
                         bias_buffer, activation_buffer));
  PrimitiveType input_primitive_type = fmha_config.input_type;
  switch (input_primitive_type) {
    case F16:
      return RunGpuFMHAImpl<half, half, half>(params, stream, scratch_buffer);
    case BF16:
      return RunGpuFMHAImpl<bfloat16, bfloat16, bfloat16>(params, stream,
                                                          scratch_buffer);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unimplemented fused MHA with %s", ToString(fmha_config)));
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
