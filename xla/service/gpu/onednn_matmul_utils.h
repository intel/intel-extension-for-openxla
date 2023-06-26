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

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
// #include "xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/scratch_allocator.h"
#include "xla/shape.h"
#include "xla/statusor.h"
#include "xla/stream_executor/blas.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
               se::DeviceMemoryBase rhs_buffer,
               se::DeviceMemoryBase output_buffer, se::Stream* stream,
               se::ScratchAllocator* scratch_allocator = nullptr);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_ONEDNN_MATMUL_UTILS_H_
