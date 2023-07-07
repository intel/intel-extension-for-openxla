/* Copyright (c) 2023 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/gemm_thunk.h"

#include <utility>

#include "tsl/platform/logging.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/onednn_matmul_utils.h"
#include "xla/service/gpu/scratch_allocator.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

GemmThunk::GemmThunk(ThunkInfo thunk_info, GemmConfig config,
                     const BufferAllocation::Slice& lhs_buffer,
                     const BufferAllocation::Slice& rhs_buffer,
                     const BufferAllocation::Slice& output_buffer)
    : Thunk(Kind::kGemm, thunk_info),
      config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer) {}

Status GemmThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto get_device_address = [&](const BufferAllocation::Slice& slice) {
    return params.buffer_allocations->GetDeviceAddress(slice);
  };

  se::DeviceMemoryBase lhs_data = get_device_address(lhs_buffer_);
  se::DeviceMemoryBase rhs_data = get_device_address(rhs_buffer_);
  se::DeviceMemoryBase output_data = get_device_address(output_buffer_);
  se::DeviceMemoryBase add_data;
  se::DeviceMemoryBase bias_data;

  auto& buffer_allocations = *params.buffer_allocations;
  se::OwningScratchAllocator<> scratch_allocator(
      buffer_allocations.device_ordinal(),
      buffer_allocations.memory_allocator());

  VLOG(3) << "Running GEMM thunk";
  return RunGemm(config_, lhs_data, rhs_data, add_data, output_data, bias_data,
                 params.stream, &scratch_allocator);
}

CublasLtMatmulThunk::CublasLtMatmulThunk(
    ThunkInfo thunk_info, GemmConfig config, int64_t algorithm_idx,
    BufferAllocation::Slice a_buffer, BufferAllocation::Slice b_buffer,
    BufferAllocation::Slice c_buffer, BufferAllocation::Slice d_buffer,
    BufferAllocation::Slice bias_buffer, BufferAllocation::Slice aux_buffer,
    BufferAllocation::Slice a_scale, BufferAllocation::Slice b_scale,
    BufferAllocation::Slice c_scale, BufferAllocation::Slice d_scale,
    BufferAllocation::Slice d_amax)
    : Thunk(Kind::kCublasLtMatmul, thunk_info),
      config_(std::move(config)),
      algorithm_idx_(algorithm_idx),
      a_buffer_(a_buffer),
      b_buffer_(b_buffer),
      c_buffer_(c_buffer),
      d_buffer_(d_buffer),
      bias_buffer_(bias_buffer),
      aux_buffer_(aux_buffer),
      a_scale_buffer_(a_scale),
      b_scale_buffer_(b_scale),
      c_scale_buffer_(c_scale),
      d_scale_buffer_(d_scale),
      d_amax_buffer_(d_amax) {}

Status CublasLtMatmulThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "Running cublas_lt matmul thunk";
  const BufferAllocations& allocs = *params.buffer_allocations;

  se::DeviceMemoryBase a, b, c, d;
  if (a_buffer_.allocation() != nullptr) {
    a = allocs.GetDeviceAddress(a_buffer_);
  }
  if (b_buffer_.allocation() != nullptr) {
    b = allocs.GetDeviceAddress(b_buffer_);
  }
  if (c_buffer_.allocation() != nullptr) {
    c = allocs.GetDeviceAddress(c_buffer_);
  }
  if (d_buffer_.allocation() != nullptr) {
    d = allocs.GetDeviceAddress(d_buffer_);
  }

  se::DeviceMemoryBase bias, a_scale, b_scale, c_scale, d_scale, d_amax;
  if (bias_buffer_.allocation() != nullptr) {
    bias = allocs.GetDeviceAddress(bias_buffer_);
  }

  se::OwningScratchAllocator<> scratch_allocator(allocs.device_ordinal(),
                                                 allocs.memory_allocator());
  return RunGemm(config_, a, b, c, d, bias, params.stream, &scratch_allocator);
}

}  // namespace gpu
}  // namespace xla
