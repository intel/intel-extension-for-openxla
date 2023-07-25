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

#include "xla/service/gpu/ccl_all_gather_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/ccl_ops.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/sycl/sycl_stream.h"

namespace xla {
namespace gpu {
using mlir::lmhlo_gpu::AllGatherStartOp;

namespace impl {
template <typename OpT>
CclAllGatherConfig GetCclAllGatherConfig(OpT op) {
  CclAllGatherConfig config;
  config.config = GetCclCollectiveConfigForMlir(op, op.getUseGlobalDeviceIds());
  return config;
}
Status CheckImplementable(AllGatherStartOp op) {
  TF_RETURN_IF_ERROR(CclCollectiveThunk::CheckImplementable());
  for (mlir::Value operand : op.getInputs()) {
    TF_RETURN_IF_ERROR(IsValidOperand(operand, Thunk::kNcclAllGather));
    Shape shape = GetShape(operand);
    if (!ShapeUtil::IsEffectivelyMostMajorDimension(
            shape, op.getAllGatherDimension())) {
      return tsl::errors::Unimplemented(absl::StrFormat(
          "all-gather dim %u is not the most major in input shape %s",
          op.getAllGatherDimension(), shape.ToString(/*print_layout=*/true)));
    }
  }
  return OkStatus();
}

}  // namespace impl

Status RunAllGather(std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                    ncclComm_t comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-gather from device ordinal: " << device_ordinal;

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  for (size_t i = 0; i < buffers.size(); ++i) {
    DeviceBufferPair& buffer = buffers[i];
    const void* send_buffer = buffer.source_buffer.opaque();
    void* recv_buffer = buffer.destination_buffer.opaque();

    PrimitiveType element_type = buffer.element_type;
    int element_count = buffer.element_count *
                        (primitive_util::IsComplexType(element_type) ? 2 : 1);

    VLOG(3) << absl::StreamFormat(
        "Calling ncclAllGather(send_buffer=%p, recv_buffer=%p, sendcount=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, element_count, static_cast<const void*>(comm),
        gpu_stream);

    sycl_allgather(send_buffer, recv_buffer, element_count, element_type,
                   gpu_stream, comm);
  }

  VLOG(3) << "Done performing all-gather for ordinal: " << device_ordinal;
  return OkStatus();
}

CclAllGatherStartThunk::CclAllGatherStartThunk(
    ThunkInfo thunk_info, AllGatherStartOp op,
    std::vector<CclCollectiveThunk::Buffer> buffers)
    : CclCollectiveThunk(Thunk::kNcclAllGatherStart, thunk_info,
                         op.getIsSync()),
      config_(impl::GetCclAllGatherConfig(op)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

/*static*/ Status CclAllGatherStartThunk::CheckImplementable(
    AllGatherStartOp op, int64_t replica_count, int64_t partition_count) {
  return AddOpDescription<CclAllGatherStartThunk>(
      impl::CheckImplementable(op), op, replica_count, partition_count);
}

/*static*/ bool CclAllGatherStartThunk::IsDegenerate(
    mlir::lmhlo_gpu::AllGatherStartOp op, int64_t replica_count,
    int64_t partition_count) {
  return impl::GetCclAllGatherConfig(op).config.IsDegenerate(replica_count,
                                                             partition_count);
}

/*static*/ CollectiveOpGroupMode CclAllGatherStartThunk::GetGroupMode(
    mlir::lmhlo_gpu::AllGatherStartOp op) {
  return impl::GetCclAllGatherConfig(op).config.group_mode;
}

Status CclAllGatherStartThunk::RunCclCollective(const ExecuteParams& params,
                                                se::Stream& stream,
                                                ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return xla::gpu::RunAllGather(device_buffers, stream, comm);
}

}  // namespace gpu
}  // namespace xla
