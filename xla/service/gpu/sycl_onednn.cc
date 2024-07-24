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

#include "xla/service/gpu/sycl_onednn.h"

#include <regex>

namespace xla {
namespace gpu {

absl::Status RunGpuConvCustomCall(
    se::Stream* stream, se::ScratchAllocator* scratch_allocator,
    std::vector<ffi::BufferBase>& operand_se_buffers,
    ffi::BufferBase& result_buffer, const ffi::Dictionary& dict,
    CudnnConvKind conv_kind) {
  TF_ASSIGN_OR_RETURN(auto conv_primitive,
                      GetOrCreateOneDnnConvPrimitive(
                          stream, dict, operand_se_buffers, result_buffer,
                          scratch_allocator, conv_kind));
  TF_RETURN_IF_ERROR(RunGpuConv(conv_primitive, dict,
                                absl::MakeSpan(operand_se_buffers),
                                result_buffer, conv_kind));
  return absl::OkStatus();
}

absl::Status RunGemmCustomCall(ffi::BufferBase* lhs, ffi::BufferBase* rhs,
                               ffi::BufferBase* add, ffi::BufferBase* output,
                               ffi::BufferBase* bias, se::Stream* stream,
                               const ffi::Dictionary& dict,
                               SYCLGemm::GemmBackendEpilogue epilogue,
                               se::ScratchAllocator* scratch_allocator) {
  se::DeviceMemoryBase lhs_data = lhs->data;
  se::DeviceMemoryBase rhs_data = rhs->data;
  se::DeviceMemoryBase output_data = output->data;
  se::DeviceMemoryBase add_data;
  se::DeviceMemoryBase bias_data;
  if (add != nullptr) add_data = add->data;
  if (bias != nullptr) bias_data = bias->data;

  return RunGemm(dict, lhs_data, rhs_data, add_data, output_data, bias_data,
                 stream, epilogue, scratch_allocator);
}

}  // namespace gpu
}  // namespace xla