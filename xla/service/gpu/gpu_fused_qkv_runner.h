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

#ifndef XLA_SERVICE_GPU_GPU_FUSED_QKV_RUNNER_H_
#define XLA_SERVICE_GPU_GPU_FUSED_QKV_RUNNER_H_

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"

namespace xla {
namespace gpu {

// This is an interim structure to hold the parameters to construct a
// GpufQKVConfig.
// Struct to describe properties of a FQKV without being tied to specific
// IR. Will be used to help build FQKV thunks from either XLA HLO or
// LHLO GPU dialect in MLIR.
struct GpufQKVDescriptor {
  Shape in_shape;
  Shape wei_shape;

  Shape out1_shape;
  Shape out2_shape;
  Shape out3_shape;

  // std::optional<Shape> bias_shape;
};

// Structure to describe static properties of a GPU fused QKV
struct GpufQKVConfig {
  static StatusOr<GpufQKVConfig> For(const GpufQKVDescriptor& fqkv_desc);

  MatrixLayout in_layout;
  MatrixLayout wei_layout;
  MatrixLayout out1_layout;
  MatrixLayout out2_layout;
  MatrixLayout out3_layout;

  // std::optional<se::dnn::TensorDescriptor> bias;
};

Status RunGpuFQKV(const GpufQKVConfig& fqkv_config,
                  se::DeviceMemoryBase in_buffer,
                  se::DeviceMemoryBase wei_buffer,
                  se::DeviceMemoryBase out1_buffer,
                  se::DeviceMemoryBase out2_buffer,
                  se::DeviceMemoryBase out3_buffer, se::Stream* stream);

}  // namespace gpu
}  // namespace xla
#endif  // XLA_SERVICE_GPU_GPU_FUSED_QKV_RUNNER_H_
