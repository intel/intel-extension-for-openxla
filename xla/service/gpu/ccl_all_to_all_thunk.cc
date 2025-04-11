/* Copyright (c) 2024 Intel Corporation

Copyright 2019 The OpenXLA Authors.

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

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/ccl_collective_thunk.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/runtime/ccl_api.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

namespace {

NcclAllToAllConfig GetNcclAllToAllConfig(const HloAllToAllInstruction* instr) {
  NcclAllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetNcclCollectiveConfig(instr, std::nullopt);
  config.has_split_dimension = instr->split_dimension().has_value();
  return config;
}

}  // namespace

NcclAllToAllStartThunk::NcclAllToAllStartThunk(
    ThunkInfo thunk_info, NcclApi* nccl_api,
    const HloAllToAllInstruction* instr,
    std::vector<NcclCollectiveThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllToAllStart, thunk_info, nccl_api,
                          IsSyncCollective(instr)),
      config_(GetNcclAllToAllConfig(instr)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

/*static*/ absl::Status NcclAllToAllStartThunk::CheckImplementable(
    const HloAllToAllInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  auto status = [&instr]() -> absl::Status {
    std::optional<uint64_t> split_dim = instr->split_dimension();
    for (HloInstruction* operand : instr->operands()) {
      Shape shape = operand->shape();
      TF_RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kNcclAllToAll));
      if (split_dim &&
          !ShapeUtil::IsEffectivelyMostMajorDimension(shape, *split_dim)) {
        return absl::UnimplementedError(absl::Substitute(
            "all-to-all split dim $0 is not the most major in input shape $1",
            *split_dim, shape.ToString(/*print_layout=*/true)));
      }
    }
    return absl::OkStatus();
  };
  return AddOpDescription<NcclAllToAllStartThunk>(
      status(), instr, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclAllToAllStartThunk::GetGroupMode(
    const HloAllToAllInstruction* instr) {
  return GetNcclAllToAllConfig(instr).config.group_mode;
}

absl::Status NcclAllToAllStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream,
    NcclApi::NcclCommHandle comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return xla::gpu::RunAllToAll(nccl_api(), config_.has_split_dimension,
                               device_buffers, stream, comm);
}

absl::Status RunAllToAll(NcclApi* nccl_api, bool has_split_dimension,
                         std::vector<DeviceBufferPair>& buffers,
                         se::Stream& stream, NcclApi::NcclCommHandle comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing " << (has_split_dimension ? "" : "non-")
          << "split all-to-all from device ordinal: " << device_ordinal;

  PrimitiveType element_type = buffers[0].element_type;
  int num_participants = CastCCLComm(comm)->nranks;
  size_t element_count = buffers[0].element_count;
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

    auto& buffer = buffers[0];
    const uint8_t* send_buffer =
        static_cast<uint8_t*>(buffer.source_buffer.opaque());
    uint8_t* recv_buffer =
        static_cast<uint8_t*>(buffer.destination_buffer.opaque());

    send_buffers.push_back(send_buffer);
    recv_buffers.push_back(recv_buffer);
  } else {
    TF_RET_CHECK(buffers.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

    for (size_t i = 0; i < buffers.size(); ++i) {
      auto& buffer = buffers[i];
      const uint8_t* send_buffer =
          static_cast<uint8_t*>(buffer.source_buffer.opaque());
      uint8_t* recv_buffer =
          static_cast<uint8_t*>(buffer.destination_buffer.opaque());

      send_buffers.push_back(send_buffer);
      recv_buffers.push_back(recv_buffer);
    }
  }

  auto ccl_api = dynamic_cast<CclApi*>(nccl_api);
  TF_RETURN_IF_ERROR(ccl_api->AllToAll(has_split_dimension, send_buffers,
                                       recv_buffers, element_count,
                                       element_type, comm, &stream));

  VLOG(3) << "Done performing all-to-all for ordinal: " << device_ordinal;
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
