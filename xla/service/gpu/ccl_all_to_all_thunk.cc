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
using mlir::lmhlo_gpu::AllToAllStartOp;

namespace impl {
template <typename OpT>
NcclAllToAllConfig GetNcclAllToAllConfig(OpT op) {
  NcclAllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetNcclCollectiveConfigForMlir(op, std::nullopt);
  config.has_split_dimension = op.getSplitDimension().has_value();
  return config;
}

Status CheckImplementable(AllToAllStartOp op) {
  TF_RETURN_IF_ERROR(NcclCollectiveThunk::CheckImplementable());
  std::optional<uint64_t> split_dim = op.getSplitDimension();
  for (mlir::Value operand : op.getInputs()) {
    TF_RETURN_IF_ERROR(IsValidOperand(operand, Thunk::kNcclAllToAll));
    Shape shape = GetShape(operand);
    if (split_dim &&
        !ShapeUtil::IsEffectivelyMostMajorDimension(shape, *split_dim)) {
      return tsl::errors::Unimplemented(
          "all-to-all split dim %u is not the most major in input shape %s",
          *split_dim, shape.ToString(/*print_layout=*/true));
    }
  }
  return OkStatus();
}

}  // namespace impl

NcclAllToAllStartThunk::NcclAllToAllStartThunk(
    ThunkInfo thunk_info, AllToAllStartOp op,
    std::vector<NcclCollectiveThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllToAllStart, thunk_info,
                          op.getIsSync()),
      config_(impl::GetNcclAllToAllConfig(op)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

/*static*/ Status NcclAllToAllStartThunk::CheckImplementable(
    AllToAllStartOp op, int64_t replica_count, int64_t partition_count) {
  return AddOpDescription<NcclAllToAllStartThunk>(
      impl::CheckImplementable(op), op, replica_count, partition_count);
}

/*static*/ bool NcclAllToAllStartThunk::IsDegenerate(
    mlir::lmhlo_gpu::AllToAllStartOp op, int64_t replica_count,
    int64_t partition_count) {
  return impl::GetNcclAllToAllConfig(op).config.IsDegenerate(replica_count,
                                                             partition_count);
}

/*static*/ CollectiveOpGroupMode NcclAllToAllStartThunk::GetGroupMode(
    mlir::lmhlo_gpu::AllToAllStartOp op) {
  return impl::GetNcclAllToAllConfig(op).config.group_mode;
}

Status NcclAllToAllStartThunk::RunNcclCollective(const ExecuteParams& params,
                                                 se::Stream& stream,
                                                 ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return xla::gpu::RunAllToAll(config_.has_split_dimension, device_buffers,
                               stream, comm);
}

NcclAllToAllDoneThunk::NcclAllToAllDoneThunk(
    ThunkInfo thunk_info, NcclCollectiveThunk::AsyncExecutor& async)
    : NcclCollectiveDoneThunk(Thunk::kNcclAllToAllDone, thunk_info, async) {}

Status RunAllToAll(bool has_split_dimension,
                   std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                   ncclComm_t comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing " << (has_split_dimension ? "" : "non-")
          << "split all-to-all from device ordinal: " << device_ordinal;

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  PrimitiveType element_type = buffers[0].element_type;
  int num_participants = comm->nranks;
  int element_count = buffers[0].element_count *
                      (primitive_util::IsComplexType(element_type) ? 2 : 1);
  std::vector<const void*> send_buffers;
  std::vector<void*> recv_buffers;

  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (has_split_dimension) {
    TF_RET_CHECK(element_count % num_participants == 0)
        << "Buffer was not an exact multiple of the number of participants.";
    TF_RET_CHECK(buffers.size() == 1)
        << "Split AllToAll only supported dimension 0 as buffer.";

    DeviceBufferPair& buffer = buffers[0];
    const uint8_t* send_buffer =
        static_cast<uint8_t*>(buffer.source_buffer.opaque());
    uint8_t* recv_buffer =
        static_cast<uint8_t*>(buffer.destination_buffer.opaque());

    send_buffers.push_back(send_buffer);
    recv_buffers.push_back(recv_buffer);
    sycl_alltoall_split(send_buffers, recv_buffers, element_count, element_type,
                        gpu_stream, comm);
  } else {
    TF_RET_CHECK(buffers.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];
      const uint8_t* send_buffer =
          static_cast<uint8_t*>(buffer.source_buffer.opaque());
      uint8_t* recv_buffer =
          static_cast<uint8_t*>(buffer.destination_buffer.opaque());

      send_buffers.push_back(send_buffer);
      recv_buffers.push_back(recv_buffer);
    }
    sycl_alltoall(send_buffers, recv_buffers, element_count, element_type,
                  gpu_stream, comm);
  }

  VLOG(3) << "Done performing all-to-all for ordinal: " << device_ordinal;
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
