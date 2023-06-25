/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSED_MHA_RUNNER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSED_MHA_RUNNER_H_

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// This is an interim structure to hold the parameters to construct a
// GpufMHAConfig.
// Struct to describe properties of a FMHA without being tied to specific
// IR. Will be used to help build FMHA thunks from either XLA HLO or
// LHLO GPU dialect in MLIR.
struct GpufMHADescriptor {
  CudnnfMHAKind kind;
  CudnnfMHABackendConfig backend_config;
  Shape lhs_bmm1_shape;
  Shape rhs_bmm1_shape;
  Shape rhs_bmm2_shape;
  Shape intermediate_lhs_bmm2_shape;
  Shape output_shape;
  DotDimensionNumbers bmm1_dnums;
  DotDimensionNumbers bmm2_dnums;

  std::optional<Shape> mask_shape;
  std::optional<Shape> bias_shape;
};

// Structure to describe static properties of a GPU fused Multi-Headed
// Attention.
struct GpufMHAConfig {
  static StatusOr<GpufMHAConfig> For(const GpufMHADescriptor& fmha_desc);
  PrimitiveType
      input_type;  // Capture the primitive type of one of the inputs of BMM1
  PrimitiveType output_type;
  CudnnfMHAKind kind;
  std::optional<double> fmha_scale;
  std::optional<double> dropout_rate;
  std::optional<int64_t> seed;

  se::dnn::AlgorithmDesc algorithm;

  // bias -> [1, num_attn_heads, q_seq_len, kv_seq_len]
  // mask -> [batch_size, 1, q_seq_len, kv_seq_len]
  se::dnn::MatmulTensorDescriptor lhs_bmm1;
  se::dnn::MatmulTensorDescriptor rhs_bmm1;
  se::dnn::MatmulTensorDescriptor rhs_bmm2;
  se::dnn::MatmulTensorDescriptor intermediate_lhs_bmm2;
  se::dnn::TensorDescriptor output;

  std::optional<se::dnn::TensorDescriptor> mask;
  std::optional<se::dnn::TensorDescriptor> bias;
};

// Implementation struct exposed for debugging and log analysis.
struct GpufMHAParams {
  static StatusOr<GpufMHAParams> For(const GpufMHAConfig& config,
                                     se::DeviceMemoryBase lhs_bmm1_buffer,
                                     se::DeviceMemoryBase rhs_bmm1_buffer,
                                     se::DeviceMemoryBase rhs_bmm2_buffer,
                                     se::DeviceMemoryBase output_buffer,
                                     se::DeviceMemoryBase mask_buffer,
                                     se::DeviceMemoryBase bias_buffer);

  const GpufMHAConfig* config;  // Not owned
  se::DeviceMemoryBase lhs_bmm1_buffer;
  se::DeviceMemoryBase rhs_bmm1_buffer;
  se::DeviceMemoryBase rhs_bmm2_buffer;
  se::DeviceMemoryBase output_buffer;
  std::optional<se::DeviceMemoryBase> mask_buffer;
  std::optional<se::DeviceMemoryBase> bias_buffer;
};

Status RunGpuFMHA(
    const GpufMHAConfig& fmha_config, se::DeviceMemoryBase lhs_bmm1_buffer,
    se::DeviceMemoryBase rhs_bmm1_buffer, se::DeviceMemoryBase rhs_bmm2_buffer,
    se::DeviceMemoryBase output_buffer, se::DeviceMemoryBase scratch_buffer,
    se::DeviceMemoryBase mask_buffer, se::DeviceMemoryBase bias_buffer,
    se::Stream* stream);

}  // namespace gpu
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSED_MHA_RUNNER_H_
