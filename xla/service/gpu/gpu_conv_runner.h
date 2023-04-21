/* Copyright (c) 2023 Intel Corporation

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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_GPU_GPU_CONV_RUNNER_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_GPU_GPU_CONV_RUNNER_H_

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/thunk.h"
#include "xla/service/onednn_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace gpu {

typedef struct OneDnnConvPrimitive {
  dnnl::memory src_memory;
  dnnl::memory filter_memory;
  dnnl::memory dst_memory;
  dnnl::memory internal_filter_memory;
  dnnl::memory scratchpad_memory;
  dnnl::memory bias_memory;
  dnnl::convolution_forward fwd_primitive;
  dnnl::convolution_backward_data bwd_input_primitive;
  dnnl::convolution_backward_weights bwd_filter_primitive;
  dnnl::reorder filter_reorder_primitive;

  std::unordered_map<int, dnnl::memory> fwd_primitives_args;
  std::unordered_map<int, dnnl::memory> bwd_input_primitive_args;
  std::unordered_map<int, dnnl::memory> bwd_filter_primitive_args;

  std::unordered_map<int, dnnl::memory> reorder_args;

  dnnl::engine engine;
  dnnl::stream stream;
  bool has_reorder = false;
} OneDnnConvPrimitive;

struct GpuConvDescriptor {
  CudnnConvKind kind;
  CudnnConvBackendConfig backend_config;
  Shape operand0_shape;
  Shape operand1_shape;
  Shape result_shape;
  size_t scratch_size;
  Window window;
  ConvolutionDimensionNumbers dnums;
  // mlir::lmhlo_gpu::Activation activation;
  int64_t feature_group_count;
};

Status RunGpuConv(const OneDnnConvPrimitive& onednn_primitive,

                  const GpuConvDescriptor& conv_descriptor,
                  absl::Span<const se::DeviceMemoryBase> operand_buffers,
                  se::DeviceMemoryBase result_buffer,
                  const Thunk::ExecuteParams& params);

}  // namespace gpu
}  // namespace xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_GPU_GPU_CONV_RUNNER_H_
