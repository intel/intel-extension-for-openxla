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
#include "xla/service/gpu/xetla/sdp/sdp_backward.h"
#include "xla/service/gpu/xetla/sdp/sdp_forward.h"
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
                         DeviceMemoryBase bias_buffer,
                         DeviceMemoryBase scratch_memory,
                         DeviceMemoryBase activation_output, bool is_training) {
  sycl::queue* dpcpp_stream = se::gpu::AsGpuStreamValue(stream);
  std::optional<double> dropout_rate;
  if (params.config->dropout_rate) {
    dropout_rate = *params.config->dropout_rate;
    VLOG(1) << "dropout_rate: " << *dropout_rate;
    if (dropout_rate != 0) {
      return Unimplemented("Unimplemented dropout");
    }
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

  if (params.config->activation) {
    auto activation_desc = *params.config->activation;
    VLOG(1) << "activation_desc: \n" << activation_desc.ToString();
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
  auto activation_ptr = reinterpret_cast<void*>(activation_output.opaque());

  // Recalculate scale since scale attr has accuracy issue.
  if ((scale - 1.0f) > 1e-6) scale = 1.0f / sqrt(H);
  if (std::is_same_v<ElementType, bfloat16>) {
    ::gpu::xetla::fmha_forward_kernel_bf16(
        *dpcpp_stream, lhs_bmm1_ptr, rhs_bmm1_ptr, rhs_bmm2_ptr, bias_ptr,
        nullptr, 1.0f, output_ptr, activation_ptr, B, N, H, F, T, scale,
        is_training);
  } else if (std::is_same_v<ElementType, half>) {
    ::gpu::xetla::fmha_forward_kernel_fp16(
        *dpcpp_stream, lhs_bmm1_ptr, rhs_bmm1_ptr, rhs_bmm2_ptr, bias_ptr,
        nullptr, 1.0f, output_ptr, activation_ptr, B, N, H, F, T, scale,
        is_training);
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
          ? se::DeviceMemory<float>(*params.activation_buffer)
          : se::DeviceMemoryBase();
  bool is_training = params.activation_buffer.has_value() ? true : false;
  auto bias_buffer = params.bias_buffer.has_value()
                         ? se::DeviceMemory<BiasType>(*params.bias_buffer)
                         : se::DeviceMemoryBase();

  se::dnn::AlgorithmDesc algorithm = params.config->algorithm;
  absl::Status run_status = absl::OkStatus();
  switch (params.config->kind) {
    case CudnnfMHAKind::kSoftmax:
    case CudnnfMHAKind::kScaleBiasSoftmax:
      run_status = RunFusedMHA<ElementType, BiasType, OutputType>(
          params, stream, lhs_bmm1_buffer, rhs_bmm1_buffer, rhs_bmm2_buffer,
          output_buffer, bias_buffer, scratch_memory, activation_buffer,
          is_training);
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
    std::optional<se::DeviceMemoryBase> bias_buffer,
    std::optional<se::DeviceMemoryBase> activation_buffer, se::Stream* stream) {
  // Add two params just for building, what do not be used
  std::optional<se::DeviceMemoryBase> seqlen_q_buffer;
  std::optional<se::DeviceMemoryBase> seqlen_k_buffer;
  TF_ASSIGN_OR_RETURN(
      GpufMHAParams params,
      GpufMHAParams::For(fmha_config, lhs_bmm1_buffer, rhs_bmm1_buffer,
                         rhs_bmm2_buffer, output_buffer, bias_buffer,
                         activation_buffer, seqlen_q_buffer, seqlen_k_buffer));
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

namespace {
using se::DeviceMemory;
using se::DeviceMemoryBase;

template <typename ElementType, typename BiasType, typename OutputType>
absl::Status RunFusedMHABackward(
    GpufMHABackwardParams params, se::Stream* stream,
    DeviceMemory<ElementType> bmm1_grad_gemm1_rhs_buffer,
    DeviceMemory<ElementType> bmm1_grad_gemm2_rhs_buffer,
    DeviceMemory<ElementType> bmm2_grad_gemm1_lhs_buffer,
    DeviceMemory<ElementType> bmm2_grad_gemm2_rhs_buffer,
    DeviceMemory<ElementType> d_output_buffer,
    DeviceMemory<OutputType> d_bmm1_lhs_buffer,
    DeviceMemory<OutputType> d_bmm1_rhs_buffer,
    DeviceMemory<OutputType> d_bmm2_rhs_buffer, DeviceMemoryBase d_s_buffer,
    DeviceMemoryBase d_bias_buffer, DeviceMemoryBase fwd_output_buffer,
    DeviceMemoryBase bias_buffer, DeviceMemoryBase softmax_buffer,
    DeviceMemoryBase accum_buffer, DeviceMemoryBase scratch_memory) {
  sycl::queue* dpcpp_stream = se::gpu::AsGpuStreamValue(stream);
  std::optional<double> dropout_rate;
  if (params.config->dropout_rate) {
    dropout_rate = *params.config->dropout_rate;
    VLOG(1) << "dropout_rate: " << *dropout_rate;
    if (dropout_rate != 0) {
      return Unimplemented("Unimplemented dropout");
    }
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

  auto bmm1_grad_gemm1_rhs_desc = params.config->bmm1_grad_gemm1_rhs;  // q
  auto bmm1_grad_gemm2_rhs_desc = params.config->bmm1_grad_gemm2_rhs;  // k
  auto bmm2_grad_gemm1_lhs_desc = params.config->bmm2_grad_gemm1_lhs;  // L
  auto bmm2_grad_gemm2_rhs_desc = params.config->bmm2_grad_gemm2_rhs;  // v
  auto d_output_desc = params.config->d_output;
  auto d_bmm1_lhs_desc = params.config->d_bmm1_lhs;
  auto d_bmm1_rhs_desc = params.config->d_bmm1_rhs;
  auto d_bmm2_rhs_desc = params.config->d_bmm2_rhs;

  VLOG(1) << "bmm1_grad_gemm1_rhs_desc: \n"
          << bmm1_grad_gemm1_rhs_desc.ToString();
  VLOG(1) << "bmm1_grad_gemm2_rhs_desc: \n"
          << bmm1_grad_gemm2_rhs_desc.ToString();
  VLOG(1) << "bmm2_grad_gemm1_lhs_desc: \n"
          << bmm2_grad_gemm1_lhs_desc.ToString();
  VLOG(1) << "bmm2_grad_gemm2_rhs_desc: \n"
          << bmm2_grad_gemm2_rhs_desc.ToString();
  VLOG(1) << "d_output_desc: \n" << d_output_desc.ToString();
  VLOG(1) << "d_bmm1_lhs_desc: \n" << d_bmm1_lhs_desc.ToString();
  VLOG(1) << "d_bmm1_rhs_desc: \n" << d_bmm1_rhs_desc.ToString();
  VLOG(1) << "d_bmm2_rhs_desc: \n" << d_bmm2_rhs_desc.ToString();

  if (params.config->bias) {
    auto bias_desc = *params.config->bias;
    VLOG(1) << "bias_desc: \n" << bias_desc.ToString();
  }

  if (params.config->fwd_output) {
    auto fwd_output_desc = *params.config->fwd_output;
    VLOG(1) << "fwd_output_desc: \n" << fwd_output_desc.ToString();
  }

  auto bmm1_grad_gemm1_rhs_dims =
      bmm1_grad_gemm1_rhs_desc.GetCudnnCompatibleDimensions(/*is_lhs*/ false);
  auto bmm1_grad_gemm2_rhs_dims =
      bmm1_grad_gemm2_rhs_desc.GetCudnnCompatibleDimensions(/*is_lhs*/ false);

  int rank = bmm1_grad_gemm1_rhs_dims.size();
  CHECK(rank == 4);

  auto bmm1_grad_gemm1_rhs_strides =
      bmm1_grad_gemm1_rhs_desc.GetCudnnCompatibleStrides(/*is_lhs*/ false);
  auto bmm1_grad_gemm2_rhs_strides =
      bmm1_grad_gemm2_rhs_desc.GetCudnnCompatibleStrides(/*is_lhs*/ false);
  auto bmm2_grad_gemm2_rhs_strides =
      bmm2_grad_gemm2_rhs_desc.GetCudnnCompatibleStrides(/*is_lhs*/ true);
  auto d_output_strides =
      d_output_desc.GetCudnnCompatibleStrides(/*is_lhs*/ false);

  CHECK(bmm1_grad_gemm1_rhs_strides[rank - 1] == 1);
  CHECK(bmm1_grad_gemm2_rhs_strides[rank - 1] == 1);
  CHECK(bmm2_grad_gemm2_rhs_strides[rank - 1] == 1);
  CHECK(d_output_strides[rank - 1] == 1);

  int B = bmm1_grad_gemm1_rhs_dims[0];
  int N = bmm1_grad_gemm1_rhs_dims[1];
  int F = bmm1_grad_gemm1_rhs_dims[2];
  int H = bmm1_grad_gemm1_rhs_dims[3];
  int T = bmm1_grad_gemm2_rhs_dims[2];

  auto q_ptr = reinterpret_cast<void*>(bmm1_grad_gemm1_rhs_buffer.opaque());
  auto k_ptr = reinterpret_cast<void*>(bmm1_grad_gemm2_rhs_buffer.opaque());
  auto v_ptr = reinterpret_cast<void*>(bmm2_grad_gemm2_rhs_buffer.opaque());
  auto o_ptr = reinterpret_cast<void*>(fwd_output_buffer.opaque());
  auto do_ptr = reinterpret_cast<void*>(d_output_buffer.opaque());
  auto bias_ptr = reinterpret_cast<void*>(bias_buffer.opaque());
  auto L_ptr = reinterpret_cast<void*>(bmm2_grad_gemm1_lhs_buffer.opaque());
  auto dq_ptr = reinterpret_cast<void*>(d_bmm1_lhs_buffer.opaque());
  auto dk_ptr = reinterpret_cast<void*>(d_bmm1_rhs_buffer.opaque());
  auto dv_ptr = reinterpret_cast<void*>(d_bmm2_rhs_buffer.opaque());

  auto dp_sum = reinterpret_cast<void*>(softmax_buffer.opaque());
  auto dq_accum_ptr = reinterpret_cast<void*>(accum_buffer.opaque());

  CHECK(o_ptr != nullptr);
  CHECK(dp_sum != nullptr);
  CHECK(dq_accum_ptr != nullptr);

  // // Recalculate scale since scale attr has accuracy issue.
  if ((scale - 1.0f) > 1e-6) scale = 1.0f / sqrt(H);
  if (std::is_same_v<ElementType, bfloat16>) {
    ::gpu::xetla::fmha_backward_kernel_bf16(
        *dpcpp_stream, q_ptr, k_ptr, v_ptr, o_ptr, bias_ptr, do_ptr, dp_sum,
        L_ptr, dq_ptr, dq_accum_ptr, dk_ptr, dv_ptr, B, N, H, F, T, scale);
  } else if (std::is_same_v<ElementType, half>) {
    ::gpu::xetla::fmha_backward_kernel_fp16(
        *dpcpp_stream, q_ptr, k_ptr, v_ptr, o_ptr, bias_ptr, do_ptr, dp_sum,
        L_ptr, dq_ptr, dq_accum_ptr, dk_ptr, dv_ptr, B, N, H, F, T, scale);
  } else {
    return Internal("Invalid MHA datatype");
  }

  return absl::OkStatus();
}

template <typename ElementType, typename BiasType, typename OutputType>
absl::Status RunGpuFMHABackwardImpl(const GpufMHABackwardParams& params,
                                    se::Stream* stream,
                                    se::DeviceMemoryBase scratch_memory) {
  auto bmm1_grad_gemm1_rhs_buffer =
      se::DeviceMemory<ElementType>(params.bmm1_grad_gemm1_rhs_buffer);
  auto bmm1_grad_gemm2_rhs_buffer =
      se::DeviceMemory<ElementType>(params.bmm1_grad_gemm2_rhs_buffer);
  auto bmm2_grad_gemm1_lhs_buffer =
      se::DeviceMemory<ElementType>(params.bmm2_grad_gemm1_lhs_buffer);
  auto bmm2_grad_gemm2_rhs_buffer =
      se::DeviceMemory<ElementType>(params.bmm2_grad_gemm2_rhs_buffer);
  auto d_output_buffer = se::DeviceMemory<ElementType>(params.d_output_buffer);
  auto d_bmm1_lhs_buffer =
      se::DeviceMemory<OutputType>(params.d_bmm1_lhs_buffer);
  auto d_bmm1_rhs_buffer =
      se::DeviceMemory<OutputType>(params.d_bmm1_rhs_buffer);
  auto d_bmm2_rhs_buffer =
      se::DeviceMemory<OutputType>(params.d_bmm2_rhs_buffer);

  // optional buffers
  auto d_s_buffer = params.d_s_buffer.has_value()
                        ? se::DeviceMemory<OutputType>(*params.d_s_buffer)
                        : se::DeviceMemoryBase();

  auto d_bias_buffer = params.d_bias_buffer.has_value()
                           ? se::DeviceMemory<OutputType>(*params.d_bias_buffer)
                           : se::DeviceMemoryBase();

  auto fwd_output_buffer =
      params.fwd_output_buffer.has_value()
          ? se::DeviceMemory<ElementType>(*params.fwd_output_buffer)
          : se::DeviceMemoryBase();

  auto bias_buffer = params.bias_buffer.has_value()
                         ? se::DeviceMemory<BiasType>(*params.bias_buffer)
                         : se::DeviceMemoryBase();

  auto softmax_buffer = params.softmax_buffer.has_value()
                            ? se::DeviceMemory<float>(*params.softmax_buffer)
                            : se::DeviceMemoryBase();

  auto accum_buffer = params.accum_buffer.has_value()
                          ? se::DeviceMemory<float>(*params.accum_buffer)
                          : se::DeviceMemoryBase();

  absl::Status run_status = absl::OkStatus();
  switch (params.config->kind) {
    case CudnnfMHAKind::kBackwardSoftmax:
    case CudnnfMHAKind::kBackwardScaleBiasSoftmax:
      run_status = RunFusedMHABackward<ElementType, OutputType>(
          params, stream, bmm1_grad_gemm1_rhs_buffer,
          bmm1_grad_gemm2_rhs_buffer, bmm2_grad_gemm1_lhs_buffer,
          bmm2_grad_gemm2_rhs_buffer, d_output_buffer, d_bmm1_lhs_buffer,
          d_bmm1_rhs_buffer, d_bmm2_rhs_buffer, d_s_buffer, d_bias_buffer,
          fwd_output_buffer, bias_buffer, softmax_buffer, accum_buffer,
          scratch_memory);
      break;
    default:
      return Internal("Invalid cuDNN fMHA kind");
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

absl::Status RunXetlaGpuFMHABackward(
    const GpufMHABackwardConfig& fmha_config,
    se::DeviceMemoryBase bmm1_grad_gemm1_rhs_buffer,
    se::DeviceMemoryBase bmm1_grad_gemm2_rhs_buffer,
    se::DeviceMemoryBase bmm2_grad_gemm1_lhs_buffer,
    se::DeviceMemoryBase bmm2_grad_gemm2_rhs_buffer,
    se::DeviceMemoryBase d_output_buffer, se::DeviceMemoryBase scratch_buffer,
    se::DeviceMemoryBase d_bmm1_lhs_buffer,
    se::DeviceMemoryBase d_bmm1_rhs_buffer,
    se::DeviceMemoryBase d_bmm2_rhs_buffer,
    std::optional<se::DeviceMemoryBase> d_s_buffer,
    std::optional<se::DeviceMemoryBase> d_bias_buffer,
    std::optional<se::DeviceMemoryBase> fwd_output_buffer,
    std::optional<se::DeviceMemoryBase> bias_buffer,
    std::optional<se::DeviceMemoryBase> softmax_buffer,
    std::optional<se::DeviceMemoryBase> accum_buffer, se::Stream* stream) {
  // Add two params just for building, what do not be used
  std::optional<se::DeviceMemoryBase> seqlen_q_buffer;
  std::optional<se::DeviceMemoryBase> seqlen_k_buffer;
  TF_ASSIGN_OR_RETURN(
      GpufMHABackwardParams params,
      GpufMHABackwardParams::For(
          fmha_config, bmm1_grad_gemm1_rhs_buffer, bmm1_grad_gemm2_rhs_buffer,
          bmm2_grad_gemm1_lhs_buffer, bmm2_grad_gemm2_rhs_buffer,
          d_output_buffer, d_bmm1_lhs_buffer, d_bmm1_rhs_buffer,
          d_bmm2_rhs_buffer, d_s_buffer, d_bias_buffer, fwd_output_buffer,
          bias_buffer, seqlen_q_buffer, seqlen_k_buffer, softmax_buffer,
          accum_buffer));
  PrimitiveType input_primitive_type = fmha_config.input_type;
  switch (input_primitive_type) {
    case F16:
      return RunGpuFMHABackwardImpl<half, half, half>(params, stream,
                                                      scratch_buffer);
    case BF16:
      return RunGpuFMHABackwardImpl<bfloat16, bfloat16, bfloat16>(
          params, stream, scratch_buffer);
    default:
      return Unimplemented("Unimplemented fused MHA backward");
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
