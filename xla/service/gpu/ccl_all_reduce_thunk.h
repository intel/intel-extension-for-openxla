/* Copyright (c) 2023 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_CCL_ALL_REDUCE_THUNK_H_
#define XLA_SERVICE_GPU_CCL_ALL_REDUCE_THUNK_H_

#include <memory>
#include <optional>
#include <vector>

#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/ccl_collective_thunk.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct CclAllReduceConfig {
  CclCollectiveConfig config;
  ReductionKind reduction_kind;
};

// Thunk that performs a NCCL-based All-Reduce or Reduce-Scatter among CUDA
// GPU-based replicas.
class CclAllReduceReduceScatterThunkBase : public CclCollectiveThunk {
 public:
  static std::optional<ReductionKind> MatchAllReduceComputation(
      mlir::Region& computation);

  CclAllReduceReduceScatterThunkBase(Kind kind, ThunkInfo thunk_info,
                                     CclAllReduceConfig config,
                                     std::vector<Buffer> buffers);

 protected:
  const CclCollectiveConfig& config() const override { return config_.config; }

  const CclAllReduceConfig config_;
  const std::vector<Buffer> buffers_;
};

// -----------------------------------------------------------------------------
// AllReduce thunks
// -----------------------------------------------------------------------------

class CclAllReduceThunkBase : public CclAllReduceReduceScatterThunkBase {
 public:
  using CclAllReduceReduceScatterThunkBase::CclAllReduceReduceScatterThunkBase;

 protected:
  Status RunAllReduce(const ExecuteParams& params, se::Stream& stream,
                      ncclComm_t comm);
};

class CclAllReduceThunk : public CclAllReduceThunkBase {
 public:
  CclAllReduceThunk(ThunkInfo thunk_info, mlir::lmhlo::AllReduceOp op,
                    std::vector<Buffer> buffers);

  static const char* GetName() { return "AllReduce"; }

  static bool CanImplement(mlir::lmhlo::AllReduceOp op);
  static bool IsDegenerate(mlir::lmhlo::AllReduceOp op, int64_t replica_count,
                           int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(mlir::lmhlo::AllReduceOp op);
  static constexpr bool IsAsync() { return false; }

 protected:
  Status RunCclCollective(const ExecuteParams& params,
                          ncclComm_t comm) override;
};

class CclAllReduceStartThunk : public CclAllReduceThunkBase {
 public:
  CclAllReduceStartThunk(ThunkInfo thunk_info,
                         mlir::lmhlo_gpu::AllReduceStartOp op,
                         std::vector<Buffer> buffers);

  static const char* GetName() { return "AllReduceStart"; }

  static bool CanImplement(mlir::lmhlo_gpu::AllReduceStartOp op);
  static bool IsDegenerate(mlir::lmhlo_gpu::AllReduceStartOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo_gpu::AllReduceStartOp op);
  static constexpr bool IsAsync() { return true; }

  AsyncExecutor& async_executor() { return async_; }

 protected:
  Status RunCclCollective(const ExecuteParams& params,
                          ncclComm_t comm) override;

 private:
  AsyncExecutor async_;
};

class CclAllReduceDoneThunk : public CclCollectiveDoneThunk {
 public:
  CclAllReduceDoneThunk(ThunkInfo thunk_info,
                        CclCollectiveThunk::AsyncExecutor& async)
      : CclCollectiveDoneThunk(Thunk::kNcclAllReduceDone, thunk_info, async) {}
};

// -----------------------------------------------------------------------------
// ReduceScatter thunks
// -----------------------------------------------------------------------------

class CclReduceScatterThunkBase : public CclAllReduceReduceScatterThunkBase {
 public:
  using CclAllReduceReduceScatterThunkBase::CclAllReduceReduceScatterThunkBase;

 protected:
  Status RunReduceScatter(const ExecuteParams& params, se::Stream& stream,
                          ncclComm_t comm);
};

class CclReduceScatterThunk : public CclReduceScatterThunkBase {
 public:
  CclReduceScatterThunk(ThunkInfo thunk_info, mlir::lmhlo::ReduceScatterOp op,
                        std::vector<Buffer> buffers);

  static const char* GetName() { return "ReduceScatter"; }

  // Returns whether the given instruction can be lowered to a nccl
  // reduce-scatter call.
  static bool CanImplement(mlir::lmhlo::ReduceScatterOp op);
  static bool IsDegenerate(mlir::lmhlo::ReduceScatterOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(mlir::lmhlo::ReduceScatterOp op);
  static constexpr bool IsAsync() { return false; }

 protected:
  Status RunCclCollective(const ExecuteParams& params,
                          ncclComm_t comm) override;
};

class CclReduceScatterStartThunk : public CclReduceScatterThunkBase {
 public:
  CclReduceScatterStartThunk(ThunkInfo thunk_info,
                             mlir::lmhlo_gpu::ReduceScatterStartOp op,
                             std::vector<Buffer> buffers);

  static const char* GetName() { return "ReduceScatterStart"; }

  // Returns whether the given instruction can be lowered to a nccl
  // reduce-scatter call.
  static bool CanImplement(mlir::lmhlo_gpu::ReduceScatterStartOp op);
  static bool IsDegenerate(mlir::lmhlo_gpu::ReduceScatterStartOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo_gpu::ReduceScatterStartOp op);
  static constexpr bool IsAsync() { return true; }
  AsyncExecutor& async_executor() { return async_; }

 protected:
  Status RunCclCollective(const ExecuteParams& params,
                          ncclComm_t comm) override;

 private:
  AsyncExecutor async_;
};

class CclReduceScatterDoneThunk : public CclCollectiveDoneThunk {
 public:
  CclReduceScatterDoneThunk(ThunkInfo thunk_info,
                            CclCollectiveThunk::AsyncExecutor& async)
      : CclCollectiveDoneThunk(Thunk::kNcclReduceScatterDone, thunk_info,
                               async) {}
};

// -----------------------------------------------------------------------------

Status RunAllReduce(ReductionKind reduction_kind,
                    std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                    ncclComm_t comm, bool allow_all_reduce_kernel);

Status RunReduceScatter(ReductionKind reduction_kind,
                        std::vector<DeviceBufferPair>& buffers,
                        se::Stream& stream, ncclComm_t comm);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CCL_ALL_REDUCE_THUNK_H_
