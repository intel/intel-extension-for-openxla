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

#include "xla/service/gpu/ccl_all_reduce_thunk.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "xla/layout_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/ccl_collective_thunk.h"
#include "xla/service/gpu/ccl_ops.h"
#include "xla/status.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/sycl/sycl_stream.h"
#include "xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
using mlir::lmhlo_gpu::AllReduceStartOp;
using mlir::lmhlo_gpu::ReduceScatterStartOp;

Status RunAllReduce(ReductionKind reduction_kind,
                    std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                    ncclComm_t comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-reduce from device ordinal: " << device_ordinal;

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  int buffer_size = buffers.size();
  for (size_t i = 0; i < buffer_size; ++i) {
    DeviceBufferPair& buffer = buffers[i];
    const void* send_buffer = buffer.source_buffer.opaque();
    void* recv_buffer = buffer.destination_buffer.opaque();

    PrimitiveType element_type = buffer.element_type;
    int element_count = buffer.element_count *
                        (primitive_util::IsComplexType(element_type) ? 2 : 1);

    VLOG(1) << absl::StreamFormat(
        "Calling ccl::allreduce(send_buffer=%p, recv_buffer=%p, count=%d, "
        "comm=%p, stream=%p, tid=%d)",
        send_buffer, recv_buffer, element_count, static_cast<const void*>(comm),
        gpu_stream, std::hash<std::thread::id>{}(std::this_thread::get_id()));

    sycl_allreduce(send_buffer, recv_buffer, element_count, element_type,
                   reduction_kind, gpu_stream, comm, i, buffer_size);
  }

  return OkStatus();
}

namespace {

// Generally, the reduction op should be the only operation in the block, except
// the terminator. However, if the type is bf16, the `FloatNormalization`
// pass will have converted the op to float32 and added type conversions.
// TODO(cjfj): Can we prevent the bf16 conversion for this computation?
StatusOr<mlir::Operation*> FindReductionOp(mlir::Block& block) {
  TF_RET_CHECK(block.getNumArguments() == 2);
  mlir::Operation* terminator = block.getTerminator();
  TF_RET_CHECK(terminator);
  TF_RET_CHECK(terminator->getNumOperands() == 1);
  mlir::Value result = terminator->getOperand(0);
  TF_RET_CHECK(block.getArgument(0).getType() == result.getType());
  TF_RET_CHECK(block.getArgument(1).getType() == result.getType());

  mlir::Operation* result_op = result.getDefiningOp();
  TF_RET_CHECK(result_op);

  // In the bf16 case, the type conversions and op might be fused.
  if (mlir::isa<mlir::mhlo::FusionOp>(result_op)) {
    return FindReductionOp(result_op->getRegion(0).front());
  }

  // Standard case.
  if (absl::c_is_permutation(result_op->getOperands(), block.getArguments())) {
    return result_op;
  }

  // bf16 case.
  TF_RET_CHECK(mlir::isa<mlir::mhlo::ConvertOp>(result_op));
  TF_RET_CHECK(result_op->getNumOperands() == 1);
  mlir::Operation* reduction_op = result_op->getOperand(0).getDefiningOp();
  TF_RET_CHECK(reduction_op);
  TF_RET_CHECK(reduction_op->getNumOperands() == 2);
  mlir::Value operand0 = reduction_op->getOperand(0);
  mlir::Value operand1 = reduction_op->getOperand(1);
  auto operand0_op = operand0.getDefiningOp<mlir::mhlo::ConvertOp>();
  auto operand1_op = operand1.getDefiningOp<mlir::mhlo::ConvertOp>();
  TF_RET_CHECK(operand0_op);
  TF_RET_CHECK(operand1_op);
  TF_RET_CHECK(operand0_op->getNumOperands() == 1);
  TF_RET_CHECK(operand1_op->getNumOperands() == 1);
  std::array<mlir::Value, 2> operands{operand0_op->getOperand(0),
                                      operand1_op->getOperand(0)};
  TF_RET_CHECK(absl::c_is_permutation(operands, block.getArguments()));
  return reduction_op;
}

}  // namespace

namespace impl {

template <typename OpT>
Status CheckImplementable(OpT op, Thunk::Kind reduction_op) {
  TF_RETURN_IF_ERROR(CclCollectiveThunk::CheckImplementable());
  for (mlir::Value operand : op.getInputs()) {
    TF_RETURN_IF_ERROR(IsValidOperand(operand, reduction_op));
  }
  if (!CclAllReduceReduceScatterThunkBase::MatchAllReduceComputation(
           op.getComputation())
           .has_value()) {
    return tsl::errors::Unimplemented("Unrecognized reduction computation");
  }
  return OkStatus();
}

template <typename OpT>
CclAllReduceConfig GetCclAllReduceConfig(OpT op) {
  std::optional<ReductionKind> reduction_kind =
      CclAllReduceReduceScatterThunkBase::MatchAllReduceComputation(
          op.getComputation());
  CHECK(reduction_kind.has_value());

  CclAllReduceConfig config;
  config.config = GetCclCollectiveConfigForMlir(op, op.getUseGlobalDeviceIds());
  config.reduction_kind = *reduction_kind;
  return config;
}

template <typename OpT>
bool IsDegenerate(OpT op, int64_t replica_count, int64_t partition_count) {
  return GetCclCollectiveConfigForMlir(op, op.getUseGlobalDeviceIds())
      .IsDegenerate(replica_count, partition_count);
}

template <typename OpT>
CollectiveOpGroupMode GetGroupMode(OpT op) {
  return GetCclAllReduceConfig(op).config.group_mode;
}

}  // namespace impl

std::optional<ReductionKind>
CclAllReduceReduceScatterThunkBase::MatchAllReduceComputation(
    mlir::Region& computation) {
  mlir::Block& block = computation.front();
  StatusOr<mlir::Operation*> reduction_op = FindReductionOp(block);
  if (!reduction_op.ok()) return std::nullopt;
  StatusOr<HloOpcode> opcode = MhloToHloOpcode(*reduction_op);
  if (!opcode.ok()) return std::nullopt;
  // Match the operation to a reduction kind. We can represent and/or of pred as
  // min/max. This works because pred is stored as an 8-bit int of value 0 or 1.
  PrimitiveType type =
      TypeToShape(block.getArgument(0).getType()).element_type();
  if (type == PRED) {
    switch (opcode.value()) {
      case HloOpcode::kAnd:
        return ReductionKind::MIN;
      case HloOpcode::kOr:
        return ReductionKind::MAX;
      default:
        return std::nullopt;
    }
  } else if (primitive_util::IsComplexType(type)) {
    // Only addition is supported for complex types.
    if (*opcode == HloOpcode::kAdd) {
      return ReductionKind::SUM;
    } else {
      return std::nullopt;
    }
  } else {
    switch (*opcode) {
      case HloOpcode::kAdd:
        return ReductionKind::SUM;
      case HloOpcode::kMultiply:
        return ReductionKind::PRODUCT;
      case HloOpcode::kMaximum:
        return ReductionKind::MAX;
      case HloOpcode::kMinimum:
        return ReductionKind::MIN;
      default:
        return std::nullopt;
    }
  }
}

CclAllReduceReduceScatterThunkBase::CclAllReduceReduceScatterThunkBase(
    Thunk::Kind kind, ThunkInfo thunk_info, CclAllReduceConfig config,
    std::vector<Buffer> buffers, bool is_sync)
    : CclCollectiveThunk(kind, thunk_info, is_sync),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

Status CclAllReduceThunkBase::RunAllReduce(const ExecuteParams& params,
                                           se::Stream& stream,
                                           ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return ::xla::gpu::RunAllReduce(config_.reduction_kind, device_buffers,
                                  stream, comm);
}

CclAllReduceStartThunk::CclAllReduceStartThunk(ThunkInfo thunk_info,
                                               AllReduceStartOp op,
                                               std::vector<Buffer> buffers)
    : CclAllReduceReduceScatterThunkBase(Thunk::kNcclAllReduceStart, thunk_info,
                                         impl::GetCclAllReduceConfig(op),
                                         std::move(buffers), op.getIsSync()) {}

Status CclAllReduceStartThunk::CheckImplementable(AllReduceStartOp op,
                                                  int64_t replica_count,
                                                  int64_t partition_count) {
  return AddOpDescription<CclAllReduceStartThunk>(
      impl::CheckImplementable(op, Thunk::kNcclAllReduceStart), op,
      replica_count, partition_count);
}

bool CclAllReduceStartThunk::IsDegenerate(mlir::lmhlo_gpu::AllReduceStartOp op,
                                          int64_t replica_count,
                                          int64_t partition_count) {
  return impl::IsDegenerate(op, replica_count, partition_count);
}

CollectiveOpGroupMode CclAllReduceStartThunk::GetGroupMode(
    mlir::lmhlo_gpu::AllReduceStartOp op) {
  return impl::GetGroupMode(op);
}

Status CclAllReduceStartThunk::RunCclCollective(const ExecuteParams& params,
                                                se::Stream& stream,
                                                ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return ::xla::gpu::RunAllReduce(config_.reduction_kind, device_buffers,
                                  stream, comm);
}

Status CclReduceScatterThunkBase::RunReduceScatter(const ExecuteParams& params,
                                                   se::Stream& stream,
                                                   ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return ::xla::gpu::RunReduceScatter(config_.reduction_kind, device_buffers,
                                      stream, comm);
}

CclReduceScatterStartThunk::CclReduceScatterStartThunk(
    ThunkInfo thunk_info, ReduceScatterStartOp op,
    std::vector<CclCollectiveThunk::Buffer> buffers)
    : CclAllReduceReduceScatterThunkBase(
          Thunk::kNcclReduceScatterStart, thunk_info,
          impl::GetCclAllReduceConfig(op), std::move(buffers), op.getIsSync()) {
}

/*static*/ Status CclReduceScatterStartThunk::CheckImplementable(
    ReduceScatterStartOp op, int64_t replica_count, int64_t partition_count) {
  return AddOpDescription<CclReduceScatterStartThunk>(
      impl::CheckImplementable(op, Thunk::kNcclReduceScatterStart), op,
      replica_count, partition_count);
}

/*static*/ bool CclReduceScatterStartThunk::IsDegenerate(
    mlir::lmhlo_gpu::ReduceScatterStartOp op, int64_t replica_count,
    int64_t partition_count) {
  return impl::IsDegenerate(op, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode CclReduceScatterStartThunk::GetGroupMode(
    mlir::lmhlo_gpu::ReduceScatterStartOp op) {
  return impl::GetGroupMode(op);
}

Status CclReduceScatterStartThunk::RunCclCollective(const ExecuteParams& params,
                                                    se::Stream& stream,
                                                    ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return ::xla::gpu::RunReduceScatter(config_.reduction_kind, device_buffers,
                                      stream, comm);
}

Status RunReduceScatter(ReductionKind reduction_kind,
                        std::vector<DeviceBufferPair>& buffers,
                        se::Stream& stream, ncclComm_t comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing reduce-scatter from device ordinal: "
          << device_ordinal;

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);
  int num_participants = comm->nranks;

  int buffer_size = buffers.size();
  for (size_t i = 0; i < buffer_size; ++i) {
    DeviceBufferPair& buffer = buffers[i];
    const void* send_buffer = buffer.source_buffer.opaque();
    void* recv_buffer = buffer.destination_buffer.opaque();

    PrimitiveType element_type = buffer.element_type;
    int element_count = buffer.element_count *
                        (primitive_util::IsComplexType(element_type) ? 2 : 1);

    // buffer.element_count is the source buffers element count. For
    // ncclReduceScatter, we need the destination buffers element count.
    TF_RET_CHECK(element_count % num_participants == 0)
        << "Source buffer was not an exact multiple of the number of "
           "participants.";

    int64_t recv_count = element_count / num_participants;
    VLOG(3) << absl::StreamFormat(
        "Calling ncclReduceScatter(send_buffer=%p, recv_buffer=%p, "
        "recvcount=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, recv_count, static_cast<const void*>(comm),
        gpu_stream);
    sycl_reduce_scatter(send_buffer, recv_buffer, recv_count, element_type,
                        reduction_kind, gpu_stream, comm, i, buffer_size);
  }

  VLOG(3) << "Done performing reduce-scatter for ordinal: " << device_ordinal;
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
