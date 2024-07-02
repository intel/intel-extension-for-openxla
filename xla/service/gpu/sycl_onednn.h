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

#ifndef XLA_SERVICE_GPU_SYCL_ONEDNN_H_
#define XLA_SERVICE_GPU_SYCL_ONEDNN_H_

#include "xla/service/gpu/onednn_gpu_conv_runner.h"
#include "xla/service/gpu/onednn_matmul_utils.h"

namespace xla {
namespace gpu {
absl::Status RunGpuConvCustomCall(
    se::Stream* stream, se::ScratchAllocator* scratch_allocator,
    std::vector<ffi::BufferBase>& operand_se_buffers,
    ffi::BufferBase& result_buffer, const ffi::Dictionary& dict,
    CudnnConvKind conv_kind);

absl::Status RunGemmCustomCall(
    ffi::BufferBase* lhs, ffi::BufferBase* rhs,
    ffi::BufferBase* add, ffi::BufferBase* output,
    ffi::BufferBase* bias, se::Stream* stream,
    const ffi::Dictionary& dict,
    SYCLGemm::GemmBackendEpilogue epilogue,
    se::ScratchAllocator* scratch_allocator = nullptr);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_SYCL_ONEDNN_H_