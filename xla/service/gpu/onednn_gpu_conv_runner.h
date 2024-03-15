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

#ifndef XLA_SERVICE_GPU_ONEDNN_GPU_CONV_RUNNER_H_
#define XLA_SERVICE_GPU_ONEDNN_GPU_CONV_RUNNER_H_

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_conv_runner.h"
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

absl::StatusOr<OneDnnConvPrimitive> GetOrCreateOneDnnConvPrimitive(
    se::Stream*, const GpuConvDescriptor& descriptor,
    const std::vector<se::DeviceMemoryBase>& operand_se_buffers,
    const se::DeviceMemoryBase& result_buffer,
    const Thunk::ExecuteParams& params,
    se::ScratchAllocator* scratch_allocator);

absl::Status RunGpuConv(const OneDnnConvPrimitive& onednn_primitive,
                  const GpuConvDescriptor& conv_descriptor,
                  absl::Span<const se::DeviceMemoryBase> operand_buffers,
                  se::DeviceMemoryBase result_buffer,
                  const Thunk::ExecuteParams& params);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_ONEDNN_GPU_CONV_RUNNER_H_
