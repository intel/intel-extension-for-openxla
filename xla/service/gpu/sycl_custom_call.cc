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
    se::Stream* stream, se::OwningScratchAllocator<>* scratch_allocator,
    std::vector<ffi::BufferBase>& operand_se_buffers,
    ffi::BufferBase result_buffer, const ffi::Dictionary dict,
    CudnnConvKind conv_kind) {
  return RunGpuConvCustomCall(stream, scratch_allocator, operand_se_buffers,
                              result_buffer, dict, conv_kind);
}

// input  + filter => output
static absl::Status SyclConvolutionForward(
    se::Stream* stream, se::OwningScratchAllocator<> scratch_allocator,
    ffi::RemainingArgs args, ffi::Dictionary dict) {
  std::vector<ffi::BufferBase> operand_se_buffers;
  operand_se_buffers.reserve(2);
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(0));
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(1));
  CudnnConvKind conv_kind = CudnnConvKind::kForward;
  return SyclConvolutionBase(stream, &scratch_allocator, operand_se_buffers,
                             *args.get<ffi::BufferBase>(2), dict, conv_kind);
}

// input  + output => filter
static absl::Status SyclConvolutionBackwardFilter(
    se::Stream* stream, se::OwningScratchAllocator<> scratch_allocator,
    ffi::RemainingArgs args, ffi::Dictionary dict) {
  std::vector<ffi::BufferBase> operand_se_buffers;
  operand_se_buffers.reserve(2);
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(0));
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(1));
  CudnnConvKind conv_kind = CudnnConvKind::kBackwardFilter;
  return SyclConvolutionBase(stream, &scratch_allocator, operand_se_buffers,
                             *args.get<ffi::BufferBase>(2), dict, conv_kind);
}

// filter  + output => input
static absl::Status SyclConvolutionBackwardInput(
    se::Stream* stream, se::OwningScratchAllocator<> scratch_allocator,
    ffi::RemainingArgs args, ffi::Dictionary dict) {
  std::vector<ffi::BufferBase> operand_se_buffers;
  operand_se_buffers.reserve(2);
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(0));
  operand_se_buffers.push_back(*args.get<ffi::BufferBase>(1));
  CudnnConvKind conv_kind = CudnnConvKind::kBackwardInput;
  return SyclConvolutionBase(stream, &scratch_allocator, operand_se_buffers,
                             *args.get<ffi::BufferBase>(2), dict, conv_kind);
}

// activation(conv(input, filter) + broadcast(bias) + (optionally) side_input)
// => output
static absl::Status SyclConvolutionBiasActivationForward(
    se::Stream* stream, se::OwningScratchAllocator<> scratch_allocator,
    ffi::RemainingArgs args, ffi::Dictionary dict) {
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
    TF_RETURN_IF_ERROR(stream->MemcpyD2D(
        &Y.data, S.data,
        absl::c_accumulate(S.dimensions, 1.0, std::multiplies<int64_t>()) *
            primitive_util::ByteWidth(S.dtype)));
  }
  CudnnConvKind conv_kind = CudnnConvKind::kForwardActivation;
  return SyclConvolutionBase(stream, &scratch_allocator, operand_se_buffers, Y,
                             dict, conv_kind);
}

XLA_FFI_DEFINE_HANDLER(kSyclConvolutionForward, SyclConvolutionForward,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::ScratchAllocator>()
                           .RemainingArgs()
                           .Attrs());

XLA_FFI_DEFINE_HANDLER(kSyclConvolutionBackwardFilter,
                       SyclConvolutionBackwardFilter,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::ScratchAllocator>()
                           .RemainingArgs()
                           .Attrs());

XLA_FFI_DEFINE_HANDLER(kSyclConvolutionBackwardInput,
                       SyclConvolutionBackwardInput,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::ScratchAllocator>()
                           .RemainingArgs()
                           .Attrs());

XLA_FFI_DEFINE_HANDLER(kSyclConvolutionBiasActivationForward,
                       SyclConvolutionBiasActivationForward,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::ScratchAllocator>()
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

static absl::Status SyclGemm(se::Stream* stream,
                             se::OwningScratchAllocator<> scratch_allocator,
                             ffi::RemainingArgs args, ffi::Dictionary dict) {
  ffi::BufferBase lhs = *args.get<ffi::BufferBase>(0);
  ffi::BufferBase rhs = *args.get<ffi::BufferBase>(1);
  ffi::BufferBase output = *args.get<ffi::BufferBase>(2);
  return RunGemmCustomCall(&lhs, &rhs, /*add*/ nullptr, &output,
                           /*bias*/ nullptr, stream, dict,
                           SYCLGemm::GemmBackendEpilogue::DEFAULT,
                           &scratch_allocator);
}

static absl::Status SyclLtMatmul(se::Stream* stream,
                                 se::OwningScratchAllocator<> scratch_allocator,
                                 ffi::RemainingArgs args,
                                 ffi::Dictionary dict) {
  int32_t epilogue = *dict.get<int32_t>("epilogue");
  auto epilogue_cuda =
      static_cast<xla::gpu::GemmBackendConfig_Epilogue>(epilogue);
  TF_ASSIGN_OR_RETURN(SYCLGemm::GemmBackendEpilogue epilogue_sycl,
                      SYCLGemm::AsSYCLEpilogue(epilogue_cuda));

  TF_ASSIGN_OR_RETURN(bool has_vector_bias,
                      SYCLGemm::EpilogueAddsVectorBias(epilogue_sycl));

  bool has_matrix_bias = *dict.get<float>("beta") != 0;
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
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::ScratchAllocator>()
                           .RemainingArgs()
                           .Attrs());

XLA_FFI_DEFINE_HANDLER(kSyclLtMatmul, SyclLtMatmul,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::ScratchAllocator>()
                           .RemainingArgs()
                           .Attrs());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__cublas$gemm", PLATFORM,
                         kSyclGemm);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__cublas$lt$matmul", PLATFORM,
                         kSyclLtMatmul);

}  // namespace gpu
}  // namespace xla
