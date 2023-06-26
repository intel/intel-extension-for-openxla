/* Copyright (c) 2023 Intel Corporation

Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/ccl_collective_thunk.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct CclCollectivePermuteConfig {
  // During a collective permute, every node optionally sends its data to
  // another node (including possibly itself) and received data from another
  // node. For each node, remember who it receives data from (source) and who
  // it send data to (target). Either are optional.
  struct SourceTargetMapEntry {
    std::optional<int64_t> source;
    std::optional<int64_t> target;
  };

  using IdToSourceTargetMap =
      absl::flat_hash_map<int64_t, SourceTargetMapEntry>;

  // Returns the source and target ID corresponding to the given ID (these IDs
  // are replica_ids for cross replica permute or partition_ids for cross
  // partition permute). The source ID is the id which will send data to this
  // ID and the target ID is the id to which this ID will send its data. Either
  // can be optional.
  static SourceTargetMapEntry GetSourceTarget(
      const IdToSourceTargetMap& id_to_source_target, int64_t id) {
    auto it = id_to_source_target.find(id);
    if (it != id_to_source_target.end()) return it->second;
    return SourceTargetMapEntry{};
  }

  CclCollectiveConfig config;
  IdToSourceTargetMap id_to_source_target;
};

// Thunk that performs a NCCL-based collective permute.
class CclCollectivePermuteThunkBase : public CclCollectiveThunk {
 public:
  CclCollectivePermuteThunkBase(Kind kind, ThunkInfo thunk_info,
                                CclCollectivePermuteConfig config,
                                const Buffer& buffer);

 protected:
  Status RunCollectivePermute(const ExecuteParams& params, se::Stream& stream,
                              ncclComm_t comm);

  const CclCollectiveConfig& config() const override { return config_.config; }

 private:
  const CclCollectivePermuteConfig config_;
  const Buffer buffer_;
};

class CclCollectivePermuteThunk : public CclCollectivePermuteThunkBase {
 public:
  static CclCollectivePermuteConfig GetCclCollectivePermuteConfig(
      mlir::lmhlo::CollectivePermuteOp op, int64_t replica_count,
      int64_t partition_count);

  static bool CanImplement(mlir::lmhlo::CollectivePermuteOp op);
  static bool IsDegenerate(mlir::lmhlo::CollectivePermuteOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo::CollectivePermuteOp op);
  static const char* GetName() { return "CollectivePermute"; }
  static constexpr bool IsAsync() { return false; }

  CclCollectivePermuteThunk(ThunkInfo thunk_info,
                            mlir::lmhlo::CollectivePermuteOp op,
                            int64_t replica_count, int64_t partition_count,
                            const Buffer& buffer);

 protected:
  Status RunCclCollective(const ExecuteParams& params,
                          ncclComm_t comm) override;
};

class CclCollectivePermuteStartThunk : public CclCollectivePermuteThunkBase {
 public:
  static CclCollectivePermuteConfig GetCclCollectivePermuteConfig(
      mlir::lmhlo_gpu::CollectivePermuteStartOp op, int64_t replica_count,
      int64_t partition_count);

  static bool CanImplement(mlir::lmhlo_gpu::CollectivePermuteStartOp op);
  static bool IsDegenerate(mlir::lmhlo_gpu::CollectivePermuteStartOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo_gpu::CollectivePermuteStartOp op);
  static const char* GetName() { return "CollectivePermuteStart"; }
  static constexpr bool IsAsync() { return true; }

  CclCollectivePermuteStartThunk(ThunkInfo thunk_info,
                                 mlir::lmhlo_gpu::CollectivePermuteStartOp op,
                                 int64_t replica_count, int64_t partition_count,
                                 const Buffer& buffer);

  AsyncExecutor& async_executor() { return async_; }

 protected:
  Status RunCclCollective(const ExecuteParams& params,
                          ncclComm_t comm) override;

 private:
  AsyncExecutor async_;
};

class CclCollectivePermuteDoneThunk : public CclCollectiveDoneThunk {
 public:
  CclCollectivePermuteDoneThunk(ThunkInfo thunk_info,
                                CclCollectiveThunk::AsyncExecutor& async);
};

Status RunCollectivePermute(
    CclCollectivePermuteConfig::SourceTargetMapEntry source_target,
    DeviceBufferPair& buffer, se::Stream& stream, ncclComm_t comm,
    absl::string_view device_string, int64_t current_id);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CCL_COLLECTIVE_PERMUTE_THUNK_H_
