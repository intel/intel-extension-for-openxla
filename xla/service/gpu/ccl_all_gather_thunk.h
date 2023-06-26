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

#ifndef XLA_SERVICE_GPU_CCL_ALL_GATHER_THUNK_H_
#define XLA_SERVICE_GPU_CCL_ALL_GATHER_THUNK_H_

#include <vector>

#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/ccl_collective_thunk.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct CclAllGatherConfig {
  CclCollectiveConfig config;
};

// Base class for thunk that performs a NCCL-based All-Gather among CUDA
// GPU-based replicas.
class CclAllGatherThunkBase : public CclCollectiveThunk {
 public:
  CclAllGatherThunkBase(Kind kind, ThunkInfo thunk_info,
                        CclAllGatherConfig config, std::vector<Buffer> buffers);

 protected:
  Status RunAllGather(const ExecuteParams& params, se::Stream& stream,
                      ncclComm_t comm);
  const CclCollectiveConfig& config() const override { return config_.config; }

 private:
  const CclAllGatherConfig config_;
  const std::vector<Buffer> buffers_;
};

class CclAllGatherThunk : public CclAllGatherThunkBase {
 public:
  CclAllGatherThunk(ThunkInfo thunk_info, mlir::lmhlo::AllGatherOp op,
                    std::vector<Buffer> buffers);

  // Returns whether the given instruction can be lowered to a nccl all-gather
  // call.
  static bool CanImplement(mlir::lmhlo::AllGatherOp op);
  static const char* GetName() { return "AllGather"; }
  static bool IsDegenerate(mlir::lmhlo::AllGatherOp op, int64_t replica_count,
                           int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(mlir::lmhlo::AllGatherOp op);
  static constexpr bool IsAsync() { return false; }

 protected:
  Status RunCclCollective(const ExecuteParams& params,
                          ncclComm_t comm) override;
};

class CclAllGatherStartThunk : public CclAllGatherThunkBase {
 public:
  CclAllGatherStartThunk(ThunkInfo thunk_info,
                         mlir::lmhlo_gpu::AllGatherStartOp op,
                         std::vector<Buffer> buffers);

  static const char* GetName() { return "AllGatherStart"; }

  static bool CanImplement(mlir::lmhlo_gpu::AllGatherStartOp op);
  static bool IsDegenerate(mlir::lmhlo_gpu::AllGatherStartOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo_gpu::AllGatherStartOp op);
  static constexpr bool IsAsync() { return true; }

  AsyncExecutor& async_executor() { return async_; }

 protected:
  Status RunCclCollective(const ExecuteParams& params,
                          ncclComm_t comm) override;

 private:
  AsyncExecutor async_;
};

class CclAllGatherDoneThunk : public CclCollectiveDoneThunk {
 public:
  CclAllGatherDoneThunk(ThunkInfo thunk_info,
                        CclCollectiveThunk::AsyncExecutor& async);
};

Status RunAllGather(std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                    ncclComm_t comm);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CCL_ALL_GATHER_THUNK_H_
