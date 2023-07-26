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

#include "xla/service/gpu/fused_qkv_thunk.h"

#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "tsl/platform/logging.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

FusedQKVThunk::FusedQKVThunk(ThunkInfo thunk_info, GpufQKVConfig config,
                             const BufferAllocation::Slice& in_buffer,
                             const BufferAllocation::Slice& wei_buffer,
                             const BufferAllocation::Slice& out1_buffer,
                             const BufferAllocation::Slice& out2_buffer,
                             const BufferAllocation::Slice& out3_buffer)
    : Thunk(Kind::kFusedQKV, thunk_info),
      config_(std::move(config)),
      in_buffer_(in_buffer),
      wei_buffer_(wei_buffer),
      out1_buffer_(out1_buffer),
      out2_buffer_(out2_buffer),
      out3_buffer_(out3_buffer) {}

Status FusedQKVThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto get_device_address = [&](const BufferAllocation::Slice& slice) {
    return params.buffer_allocations->GetDeviceAddress(slice);
  };

  se::DeviceMemoryBase in_data = get_device_address(in_buffer_);
  se::DeviceMemoryBase wei_data = get_device_address(wei_buffer_);
  se::DeviceMemoryBase out1_data = get_device_address(out1_buffer_);
  se::DeviceMemoryBase out2_data = get_device_address(out2_buffer_);
  se::DeviceMemoryBase out3_data = get_device_address(out3_buffer_);
  // TODO: add bias in the future
  // se::DeviceMemoryBase bias_data;

  VLOG(1) << "Running fusedQKV thunk";
  TF_RETURN_IF_ERROR(RunGpuFQKV(config_, in_data, wei_data, out1_data,
                                out2_data, out3_data, params.stream));

  if (!params.stream->ok()) {
    return InternalError("FusedMHAThunk::ExecuteOnStream failed.");
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
