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

// Thunk that performs a NCCL-based All-Gather among CUDA GPU-based replicas.
class CclAllGatherStartThunk : public CclCollectiveThunk {
 public:
  CclAllGatherStartThunk(ThunkInfo thunk_info,
                         mlir::lmhlo_gpu::AllGatherStartOp op,
                         std::vector<Buffer> buffers);

  static const char* GetHloOpName() { return "all-gather-start"; }

  static Status CheckImplementable(mlir::lmhlo_gpu::AllGatherStartOp op,
                                   int64_t replica_count,
                                   int64_t partition_count);
  static bool IsDegenerate(mlir::lmhlo_gpu::AllGatherStartOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo_gpu::AllGatherStartOp op);

 protected:
  const CclCollectiveConfig& config() const override { return config_.config; }
  Status RunCclCollective(const ExecuteParams& params, se::Stream& stream,
                          ncclComm_t comm) override;

 private:
  const CclAllGatherConfig config_;
  const std::vector<Buffer> buffers_;
};

Status RunAllGather(std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                    ncclComm_t comm);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CCL_ALL_GATHER_THUNK_H_
