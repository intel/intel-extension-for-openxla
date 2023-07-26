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

#ifndef XLA_SERVICE_GPU_FUSED_QKV_THUNK_H_
#define XLA_SERVICE_GPU_FUSED_QKV_THUNK_H_

#include <optional>

#include "absl/container/flat_hash_map.h"
#include "tsl/platform/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_fused_qkv_runner.h"
#include "xla/service/gpu/thunk.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// This is thread-compatible.
class FusedQKVThunk : public Thunk {
 public:
  // Constructs a thunk that computes "output = (lhs <dot> rhs) * alpha" using
  // BLAS gemm (alpha is stored in the instruction GemmBackendConfig).
  FusedQKVThunk(ThunkInfo thunk_info, GpufQKVConfig config,
                const BufferAllocation::Slice& in_buffer,
                const BufferAllocation::Slice& wei_buffer,
                const BufferAllocation::Slice& out1_buffer,
                const BufferAllocation::Slice& out2_buffer,
                const BufferAllocation::Slice& out3_buffer);

  FusedQKVThunk(const FusedQKVThunk&) = delete;
  FusedQKVThunk& operator=(const FusedQKVThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const GpufQKVConfig config_;
  const BufferAllocation::Slice in_buffer_;
  const BufferAllocation::Slice wei_buffer_;
  const BufferAllocation::Slice out1_buffer_;
  const BufferAllocation::Slice out2_buffer_;
  const BufferAllocation::Slice out3_buffer_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSED_QKV_THUNK_H_
