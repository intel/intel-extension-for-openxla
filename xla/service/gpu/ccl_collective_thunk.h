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

#ifndef XLA_SERVICE_GPU_CCL_COLLECTIVE_THUNK_H_
#define XLA_SERVICE_GPU_CCL_COLLECTIVE_THUNK_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/ccl_utils.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/xla_data.pb.h"

using ncclComm_t = ccl::communicator*;

namespace xla {
namespace gpu {

class CclClique;

struct CclCollectiveConfig {
  CclCollectiveConfig();
  CclCollectiveConfig(CclCollectiveConfig&&);
  ~CclCollectiveConfig();

  CclCollectiveConfig& operator=(CclCollectiveConfig&&);

  int64_t operand_count;
  std::vector<PrimitiveType> operand_element_type;
  std::vector<ReplicaGroup> replica_groups;
  RendezvousKey::CollectiveOpKind collective_op_kind;
  int64_t op_id;
  CollectiveOpGroupMode group_mode;

  template <typename OpT>
  void SetCollectiveOpKindAndID(OpT op);
  bool IsDegenerate(int64_t replica_count, int64_t partition_count) const;
};

template <typename OpT>
void CclCollectiveConfig::SetCollectiveOpKindAndID(OpT op) {
  if (op.getChannelId()) {
    collective_op_kind = RendezvousKey::kCrossModule;
    op_id = static_cast<int64_t>(op.getChannelId()->getHandle());
  } else {
    collective_op_kind = RendezvousKey::kCrossReplica;
    mlir::ModuleOp parent = op->template getParentOfType<mlir::ModuleOp>();
    mlir::IntegerAttr unique_id =
        parent->getAttrOfType<mlir::IntegerAttr>("hlo.unique_id");
    op_id = static_cast<int64_t>(unique_id.getInt());
  }
}

template <typename OpT>
CclCollectiveConfig GetCclCollectiveConfigForMlir(
    OpT op, std::optional<bool> use_global_device_ids) {
  CclCollectiveConfig config;
  config.operand_count = op.getInputs().size();
  config.operand_element_type.reserve(config.operand_count);
  for (int i = 0; i < config.operand_count; i++) {
    const Shape shape = GetShape(op.getInputs()[i]);
    config.operand_element_type.push_back(shape.element_type());
  }
  config.replica_groups = ConvertReplicaGroups(op.getReplicaGroups()).value();
  config.SetCollectiveOpKindAndID(op);
  config.group_mode = GetCollectiveOpGroupMode(op.getChannelId().has_value(),
                                               use_global_device_ids)
                          .value();
  return config;
}

// Thunk base class for NCCL collective operations.
class CclCollectiveThunk : public Thunk {
 public:
  using Thunk::Thunk;

  struct Buffer {
    int64_t element_count;
    BufferAllocation::Slice source_buffer;
    BufferAllocation::Slice destination_buffer;
    mlir::Value source_value;
    mlir::Value destination_value;
  };

  class AsyncExecutor {
   public:
    // Executes the function on the async communications stream and records a
    // completion event.
    Status Execute(
        absl::FunctionRef<Status(const ExecuteParams&, se::Stream&, ncclComm_t)>
            fn,
        const ExecuteParams& params, ncclComm_t comm);
    // Blocks the compute stream until async communication is complete.
    Status Await(const ExecuteParams& params);

   private:
    absl::Mutex mu_;
    // Store done events (by device ordinal) for the done thunk to wait on.
    absl::flat_hash_map<int, se::Event> done_events_ ABSL_GUARDED_BY(mu_);
  };

  // Returns whether NCCL operations appear possible to perform; e.g. if we
  // haven't done a build with the CUDA compiler enabled, we can't compile the
  // NCCL header, and thus this will be false.
  //
  // When this is false, the ExecuteOnStream() call will simply return a status
  // error.
  static bool CclIsEnabled();

  // Logging support.
  static std::string GetDeviceString(const NcclExecuteParams& params);

  Status ExecuteOnStream(const ExecuteParams& params) override;

 protected:
  virtual Status RunCclCollective(const ExecuteParams& params,
                                  ncclComm_t comm) = 0;
  virtual const CclCollectiveConfig& config() const = 0;

 private:
  bool first_call_to_execute_ = true;
};

class CclCollectiveDoneThunk : public Thunk {
 public:
  CclCollectiveDoneThunk(Thunk::Kind kind, ThunkInfo thunk_info,
                         CclCollectiveThunk::AsyncExecutor& async);

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  CclCollectiveThunk::AsyncExecutor& async_;
};

// Returns if the given data type is supported by NCCL.
// Note: Keep this in sync with ToNcclDataType().
bool IsTypeSupportedByCcl(PrimitiveType element_type, Thunk::Kind reduction_op);

// TODO(hanbinyoon): Consider moving to ccl_utils.h when deprecating Thunks.
StatusOr<CclComm::Lock> LockCclComm(
    const NcclExecuteParams& params,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, int64_t op_id);

struct DeviceBufferPair {
  PrimitiveType element_type;
  int64_t element_count;
  se::DeviceMemoryBase source_buffer;
  se::DeviceMemoryBase destination_buffer;
};
StatusOr<std::vector<DeviceBufferPair>> ConvertToDeviceBuffers(
    const Thunk::ExecuteParams& params,
    const std::vector<CclCollectiveThunk::Buffer>& buffers,
    const std::vector<PrimitiveType>& element_types);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CCL_COLLECTIVE_THUNK_H_
