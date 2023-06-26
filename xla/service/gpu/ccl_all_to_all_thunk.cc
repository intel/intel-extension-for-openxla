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

#include "xla/service/gpu/ccl_all_to_all_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/ccl_collective_thunk.h"
#include "xla/service/gpu/ccl_ops.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/sycl/sycl_stream.h"

namespace xla {
namespace gpu {

namespace impl {
template <typename OpT>
CclAllToAllConfig GetCclAllToAllConfig(OpT op) {
  CclAllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetCclCollectiveConfigForMlir(op, std::nullopt);
  config.has_split_dimension = op.getSplitDimension().has_value();
  return config;
}

template <typename OpT>
bool CanImplement(OpT op) {
  return absl::c_all_of(op.getInputs(), [&op](mlir::Value operand) {
    Shape shape = GetShape(operand);
    return LayoutUtil::IsDenseArray(shape) &&
           IsTypeSupportedByCcl(shape.element_type(), Thunk::kNcclAllToAll) &&
           (!op.getSplitDimension() ||
            LayoutUtil::MinorToMajor(shape).back() == *op.getSplitDimension());
  });
}
}  // namespace impl

CclAllToAllThunkBase::CclAllToAllThunkBase(Kind kind, ThunkInfo thunk_info,
                                           CclAllToAllConfig config,
                                           std::vector<Buffer> buffers)
    : CclCollectiveThunk(kind, thunk_info),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

Status CclAllToAllThunkBase::RunAllToAll(const ExecuteParams& params,
                                         se::Stream& stream, ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return xla::gpu::RunAllToAll(config_.has_split_dimension, device_buffers,
                               stream, comm);
}

CclAllToAllThunk::CclAllToAllThunk(
    ThunkInfo thunk_info, mlir::lmhlo::AllToAllOp op,
    std::vector<CclAllToAllThunk::Buffer> buffers)
    : CclAllToAllThunkBase(Thunk::kNcclAllToAll, thunk_info,
                           impl::GetCclAllToAllConfig(op), std::move(buffers)) {
}

/*static*/ bool CclAllToAllThunk::CanImplement(mlir::lmhlo::AllToAllOp op) {
  return impl::CanImplement(op);
}

/*static*/ bool CclAllToAllThunk::IsDegenerate(mlir::lmhlo::AllToAllOp op,
                                               int64_t replica_count,
                                               int64_t partition_count) {
  return impl::GetCclAllToAllConfig(op).config.IsDegenerate(replica_count,
                                                            partition_count);
}

/*static*/ CollectiveOpGroupMode CclAllToAllThunk::GetGroupMode(
    mlir::lmhlo::AllToAllOp op) {
  return impl::GetCclAllToAllConfig(op).config.group_mode;
}

Status CclAllToAllThunk::RunCclCollective(const ExecuteParams& params,
                                          ncclComm_t comm) {
  return RunAllToAll(params, *params.stream, comm);
}

CclAllToAllStartThunk::CclAllToAllStartThunk(
    ThunkInfo thunk_info, mlir::lmhlo_gpu::AllToAllStartOp op,
    std::vector<CclAllToAllThunk::Buffer> buffers)
    : CclAllToAllThunkBase(Thunk::kNcclAllToAllStart, thunk_info,
                           impl::GetCclAllToAllConfig(op), std::move(buffers)) {
}

/*static*/ bool CclAllToAllStartThunk::CanImplement(
    mlir::lmhlo_gpu::AllToAllStartOp op) {
  return impl::CanImplement(op);
}

/*static*/ bool CclAllToAllStartThunk::IsDegenerate(
    mlir::lmhlo_gpu::AllToAllStartOp op, int64_t replica_count,
    int64_t partition_count) {
  return impl::GetCclAllToAllConfig(op).config.IsDegenerate(replica_count,
                                                            partition_count);
}

/*static*/ CollectiveOpGroupMode CclAllToAllStartThunk::GetGroupMode(
    mlir::lmhlo_gpu::AllToAllStartOp op) {
  return impl::GetCclAllToAllConfig(op).config.group_mode;
}

Status CclAllToAllStartThunk::RunCclCollective(const ExecuteParams& params,
                                               ncclComm_t comm) {
  return async_.Execute(
      [this](const ExecuteParams& params, se::Stream& stream, ncclComm_t comm) {
        return RunAllToAll(params, stream, comm);
      },
      params, comm);
}

CclAllToAllDoneThunk::CclAllToAllDoneThunk(
    ThunkInfo thunk_info, CclCollectiveThunk::AsyncExecutor& async)
    : CclCollectiveDoneThunk(Thunk::kNcclAllToAllDone, thunk_info, async) {}

Status RunAllToAll(bool has_split_dimension,
                   std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                   ncclComm_t comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-to-all from device ordinal: " << device_ordinal;

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  int num_participants = comm->nranks;
  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (has_split_dimension) {
    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];
      const uint8_t* send_buffer =
          static_cast<uint8_t*>(buffer.source_buffer.opaque());
      uint8_t* recv_buffer =
          static_cast<uint8_t*>(buffer.destination_buffer.opaque());

      PrimitiveType element_type = buffer.element_type;
      int element_count = buffers[0].element_count *
                          (primitive_util::IsComplexType(element_type) ? 2 : 1);

      TF_RET_CHECK(element_count % num_participants == 0)
          << "Buffer was not an exact multiple of the number of participants.";
      size_t chunk_elements = element_count / num_participants;
      size_t chunk_bytes = chunk_elements * ShapeUtil::ByteSizeOfPrimitiveType(
                                                buffer.element_type);

      return Unimplemented("AllToAll has_split_dimension is not supported.");
    }
  } else {
    TF_RET_CHECK(buffers.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

    std::vector<const void*> send_buffers;
    std::vector<void*> recv_buffers;
    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];
      const uint8_t* send_buffer =
          static_cast<uint8_t*>(buffer.source_buffer.opaque());
      uint8_t* recv_buffer =
          static_cast<uint8_t*>(buffer.destination_buffer.opaque());
      send_buffers.push_back(send_buffer);
      recv_buffers.push_back(recv_buffer);
    }

    PrimitiveType element_type = buffers[0].element_type;
    int element_count = buffers[0].element_count *
                        (primitive_util::IsComplexType(element_type) ? 2 : 1);

    sycl_alltoall(send_buffers, recv_buffers, element_count, element_type,
                  gpu_stream, comm);
  }

  VLOG(3) << "Done performing all-to-all for ordinal: " << device_ordinal;
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
