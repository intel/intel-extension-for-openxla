/* Copyright (c) 2024 Intel Corporation

Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_CCL_COLLECTIVE_PERMUTE_THUNK_H_
#define XLA_SERVICE_GPU_CCL_COLLECTIVE_PERMUTE_THUNK_H_

#include <cstdint>

#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/ccl_collective_thunk.h"
#include "xla/service/gpu/ccl_p2p_thunk_common.h"

namespace xla {
namespace gpu {

// Thunk that performs a NCCL-based collective permute.
class NcclCollectivePermuteStartThunk : public NcclCollectiveThunk {
 public:
  static NcclP2PConfig GetNcclP2PConfig(
      const HloCollectivePermuteInstruction* instr, int64_t replica_count,
      int64_t partition_count);

  static bool IsDegenerate(const HloCollectivePermuteInstruction* instr,
                           int64_t replica_count, int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloCollectivePermuteInstruction* instr);

  NcclCollectivePermuteStartThunk(ThunkInfo thunk_info, NcclApi* nccl_api,
                                  const HloCollectivePermuteInstruction* instr,
                                  int64_t replica_count,
                                  int64_t partition_count, const Buffer& buffer,
                                  bool p2p_memcpy_enabled);

  static const char* GetHloOpName() { return "collective-permute-start"; }

 protected:
  const NcclCollectiveConfig& config() const override { return config_.config; }
  absl::Status RunNcclCollective(const ExecuteParams& params,
                                 se::Stream& stream,
                                 NcclApi::NcclCommHandle comm) override;

 private:
  const NcclP2PConfig config_;
  const Buffer buffer_;
  bool p2p_memcpy_enabled_ = false;
};

absl::Status RunCollectivePermute(
    NcclApi* nccl_api, NcclP2PConfig::SourceTargetMapEntry source_target,
    DeviceBufferPair& buffer, se::Stream& stream, NcclApi::NcclCommHandle comm,
    absl::string_view device_string, int64_t current_id);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CCL_COLLECTIVE_PERMUTE_THUNK_H_
