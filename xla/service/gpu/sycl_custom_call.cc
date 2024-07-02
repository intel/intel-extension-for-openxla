/* Copyright (c) 2024 Intel Corporation

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

#include <string>

#include "absl/status/status.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/service/gpu/sycl_onednn.h"
#include "xla/service/onednn_util.h"
#include "xla/stream_executor/scratch_allocator.h"

#define PLATFORM "SYCL"

namespace xla {
namespace gpu {

static absl::Status SyclConvolutionBase(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ffi::BufferBase>& operand_se_buffers,
    ffi::BufferBase result_buffer, const ffi::Dictionary dict,
    CudnnConvKind conv_kind) {
  auto stream = run_options->stream();
  se::OwningScratchAllocator<2> scratch_allocator(run_options->device_ordinal(),
                                                  run_options->allocator());
  return RunGpuConvCustomCall(stream, &scratch_allocator, operand_se_buffers,
                              result_buffer, dict, conv_kind);
}

// input  + filter => output
static absl::Status SyclConvolutionForward(
    const ServiceExecutableRunOptions* run_options, ffi::RemainingArgs args,
    ffi::Dictionary dict) {
  std::vector<ffi::BufferBase> operand_se_buffers;
  operand_se_buffers.reserve(2);
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(0));
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(1));
  CudnnConvKind conv_kind = CudnnConvKind::kForward;
  return SyclConvolutionBase(run_options, operand_se_buffers,
                             *args.get<ffi::BufferBase>(2), dict, conv_kind);
}

// input  + output => filter
static absl::Status SyclConvolutionBackwardFilter(
    const ServiceExecutableRunOptions* run_options, ffi::RemainingArgs args,
    ffi::Dictionary dict) {
  std::vector<ffi::BufferBase> operand_se_buffers;
  operand_se_buffers.reserve(2);
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(0));
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(1));
  CudnnConvKind conv_kind = CudnnConvKind::kBackwardFilter;
  return SyclConvolutionBase(run_options, operand_se_buffers,
                             *args.get<ffi::BufferBase>(2), dict, conv_kind);
}

// filter  + output => input
static absl::Status SyclConvolutionBackwardInput(
    const ServiceExecutableRunOptions* run_options, ffi::RemainingArgs args,
    ffi::Dictionary dict) {
  std::vector<ffi::BufferBase> operand_se_buffers;
  operand_se_buffers.reserve(2);
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(0));
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(1));
  CudnnConvKind conv_kind = CudnnConvKind::kBackwardInput;
  return SyclConvolutionBase(run_options, operand_se_buffers,
                             *args.get<ffi::BufferBase>(2), dict, conv_kind);
}

// activation(conv(input, filter) + broadcast(bias) + (optionally) side_input)
// => output
static absl::Status SyclConvolutionBiasActivationForward(
    const ServiceExecutableRunOptions* run_options, ffi::RemainingArgs args,
    ffi::Dictionary dict) {
  std::vector<ffi::BufferBase> operand_se_buffers;
  operand_se_buffers.reserve(args.size() - 2);
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(0));  // X
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(1));  // W
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(2));  // B

  // The first and the last element in the result tuple for a convolution are
  // always the result and the scratch buffer. It may have auxiliary results in
  // addition to the main result.
  // So here need to pick the second to last buffer as the real result buffer.
  ffi::BufferBase Y = *args.get<ffi::BufferBase>(args.size() - 2);

  // SYCL: OneDNN requires inplace sum.
  if (args.size() > 3 && args.size() - 2 != 3) {
    ffi::BufferBase S = *args.get<ffi::BufferBase>(3);
    operand_se_buffers.push_back(S);
    TF_RETURN_IF_ERROR(run_options->stream()->MemcpyD2D(
        &Y.data, S.data,
        absl::c_accumulate(S.dimensions, 1.0, std::multiplies<int64_t>()) *
            primitive_util::ByteWidth(S.dtype)));
  }
  CudnnConvKind conv_kind = CudnnConvKind::kForwardActivation;
  return SyclConvolutionBase(run_options, operand_se_buffers, Y, dict,
                             conv_kind);
}

XLA_FFI_DEFINE_HANDLER(kSyclConvolutionForward, SyclConvolutionForward,
                       ffi::Ffi::Bind()
                           .Ctx<ServiceExecutableRunOptions>()
                           .RemainingArgs()
                           .Attrs());

XLA_FFI_DEFINE_HANDLER(kSyclConvolutionBackwardFilter,
                       SyclConvolutionBackwardFilter,
                       ffi::Ffi::Bind()
                           .Ctx<ServiceExecutableRunOptions>()
                           .RemainingArgs()
                           .Attrs());

XLA_FFI_DEFINE_HANDLER(kSyclConvolutionBackwardInput,
                       SyclConvolutionBackwardInput,
                       ffi::Ffi::Bind()
                           .Ctx<ServiceExecutableRunOptions>()
                           .RemainingArgs()
                           .Attrs());

XLA_FFI_DEFINE_HANDLER(kSyclConvolutionBiasActivationForward,
                       SyclConvolutionBiasActivationForward,
                       ffi::Ffi::Bind()
                           .Ctx<ServiceExecutableRunOptions>()
                           .RemainingArgs()
                           .Attrs());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__cudnn$convForward", PLATFORM,
                         kSyclConvolutionForward);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__cudnn$convBackwardFilter",
                         PLATFORM, kSyclConvolutionBackwardFilter);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__cudnn$convBackwardInput",
                         PLATFORM, kSyclConvolutionBackwardInput);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__cudnn$convBiasActivationForward", PLATFORM,
                         kSyclConvolutionBiasActivationForward);

static absl::Status SyclGemm(const ServiceExecutableRunOptions* run_options,
                             ffi::RemainingArgs args, ffi::Dictionary dict) {
  auto stream = run_options->stream();
  se::OwningScratchAllocator<2> scratch_allocator(run_options->device_ordinal(),
                                                  run_options->allocator());
  ffi::BufferBase lhs = *args.get<ffi::BufferBase>(0);
  ffi::BufferBase rhs = *args.get<ffi::BufferBase>(1);
  ffi::BufferBase output = *args.get<ffi::BufferBase>(2);
  return RunGemmCustomCall(&lhs, &rhs, /*add*/ nullptr, &output,
                           /*bias*/ nullptr, stream, dict, SYCLGemm::GemmBackendEpilogue::DEFAULT,
                           &scratch_allocator);
}

static absl::Status SyclLtMatmul(const ServiceExecutableRunOptions* run_options,
                                 ffi::RemainingArgs args,
                                 ffi::Dictionary dict) {
  auto stream = run_options->stream();
  se::OwningScratchAllocator<2> scratch_allocator(run_options->device_ordinal(),
                                                  run_options->allocator());
  int32_t epilogue = *dict.get<int32_t>("epilogue");
  auto epilogue_cuda = static_cast<xla::gpu::GemmBackendConfig_Epilogue>(epilogue);
  TF_ASSIGN_OR_RETURN(SYCLGemm::GemmBackendEpilogue epilogue_sycl,
                      SYCLGemm::AsSYCLEpilogue(epilogue_cuda));
  TF_ASSIGN_OR_RETURN(bool has_vector_bias,
                      SYCLGemm::EpilogueAddsVectorBias(epilogue_sycl));
  int64_t gemm_config_ptr = *dict.get<int64_t>("gemm_config_ptr");
  GemmConfig gemm_config = *reinterpret_cast<GemmConfig*>(gemm_config_ptr);
  bool has_matrix_bias = gemm_config.beta != 0;
  TF_ASSIGN_OR_RETURN(bool has_aux_output,
                      SYCLGemm::EpilogueHasAuxiliaryOutput(epilogue_sycl));

  ffi::BufferBase lhs = *args.get<ffi::BufferBase>(0);
  ffi::BufferBase rhs = *args.get<ffi::BufferBase>(1);
  ffi::BufferBase output;
  ffi::BufferBase add = *args.get<ffi::BufferBase>(
      2 + has_matrix_bias + has_vector_bias + (has_aux_output ? 1 : 0));
  ffi::BufferBase bias;

  if (has_matrix_bias) {
    output = *args.get<ffi::BufferBase>(2);
  } else {
    output = *args.get<ffi::BufferBase>(2 + has_matrix_bias + has_vector_bias +
                                        (has_aux_output ? 1 : 0));
  }

  if (has_vector_bias) {
    bias = *args.get<ffi::BufferBase>(has_matrix_bias ? 3 : 2);
    return RunGemmCustomCall(&lhs, &rhs, &add, &output, &bias, stream, dict,
                      epilogue_sycl, &scratch_allocator);
  } else {
    return RunGemmCustomCall(&lhs, &rhs, &add, &output, /*bias*/ nullptr,
                      stream, dict, epilogue_sycl, &scratch_allocator);
  }
}

XLA_FFI_DEFINE_HANDLER(kSyclGemm, SyclGemm,
                       ffi::Ffi::Bind()
                           .Ctx<ServiceExecutableRunOptions>()
                           .RemainingArgs()
                           .Attrs());

XLA_FFI_DEFINE_HANDLER(kSyclLtMatmul, SyclLtMatmul,
                       ffi::Ffi::Bind()
                           .Ctx<ServiceExecutableRunOptions>()
                           .RemainingArgs()
                           .Attrs());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__cublas$gemm", PLATFORM,
                         kSyclGemm);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__cublas$lt$matmul", PLATFORM,
                         kSyclLtMatmul);

}  // namespace gpu
}  // namespace xla
