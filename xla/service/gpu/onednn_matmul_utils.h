/* Copyright (c) 2023 Intel Corporation

Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_ONEDNN_MATMUL_UTILS_H_
#define XLA_SERVICE_GPU_ONEDNN_MATMUL_UTILS_H_

#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/scratch_allocator.h"

namespace xla {
namespace gpu {

namespace SYCLGemm {
enum class GemmBackendEpilogue {
  DEFAULT,
  RELU,
  GELU,
  BIAS,
  BIAS_RELU,
  BIAS_GELU,
  GELU_AUX,
  BIAS_GELU_AUX,
};

absl::StatusOr<GemmBackendEpilogue> EpilogueCast(std::string& epilogue);

absl::StatusOr<std::string> EpilogueCast(GemmBackendEpilogue epilogue);

absl::StatusOr<bool> EpilogueAddsVectorBias(GemmBackendEpilogue epilogue);

absl::StatusOr<bool> EpilogueHasAuxiliaryOutput(GemmBackendEpilogue epilogue);

absl::StatusOr<GemmBackendEpilogue> AsSYCLEpilogue(
    GemmBackendConfig_Epilogue epilogue);
}  // namespace SYCLGemm

absl::Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
                     se::DeviceMemoryBase rhs_buffer,
                     se::DeviceMemoryBase add_buffer,
                     se::DeviceMemoryBase output_buffer,
                     se::DeviceMemoryBase bias_buffer, se::Stream* stream,
                     SYCLGemm::GemmBackendEpilogue epilogue,
                     se::ScratchAllocator* scratch_allocator = nullptr);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_ONEDNN_MATMUL_UTILS_H_
