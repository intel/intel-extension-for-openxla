/* Copyright (c) 2024 Intel Corporation

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

#include "xla/pjrt/itex_pjrt_buffer.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tsl/profiler/lib/traceme.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/tf_allocator_adapter.h"

namespace xla {

StatusOr<std::unique_ptr<ITEXPjRtBuffer>> AllocateITEXDestinationBuffer(
    int device_id, PjRtDevice* device, PjRtClient* client,
    absl::Span<const int64_t> dimensions, PrimitiveType element_type,
    size_t size) {
  auto pjrt_client = dynamic_cast<PjRtStreamExecutorClient*>(client);
  auto allocator = dynamic_cast<stream_executor::MultiDeviceAdapter*>(
      pjrt_client->allocator());
  TF_ASSIGN_OR_RETURN(auto memory,
                      allocator->Allocate(device_id, size, true, 0));
  return std::make_unique<ITEXPjRtBuffer>(
      device_id, memory.Release(), dimensions, element_type, client, device);
}

ITEXPjRtBuffer::ITEXPjRtBuffer(int device_id,
                               se::DeviceMemoryBase device_memory,
                               absl::Span<const int64_t> dimensions,
                               PrimitiveType element_type, PjRtClient* client,
                               PjRtDevice* device)
    : dimensions_(dimensions.begin(), dimensions.end()),
      element_type_(element_type),
      shape_(element_type, dimensions, absl::InlinedVector<bool, 1>(),
             std::vector<Shape>()),
      client_(tensorflow::down_cast<PjRtStreamExecutorClient*>(client)),
      device_(tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)),
      buffer_(device_memory) {
  device_ordinal_ = device_id;
  allocator_ = reinterpret_cast<stream_executor::MultiDeviceAdapter*>(
      client_->allocator());
}

ITEXPjRtBuffer::~ITEXPjRtBuffer() { Delete(); }

StatusOr<std::unique_ptr<PjRtBuffer>> ITEXPjRtBuffer::CopyToDevice(
    PjRtDevice* dst_device) {
  tsl::profiler::TraceMe traceme("ITEXPjRtBuffer::CopyToDevice");
  VLOG(1) << "ITEXPjRtBuffer::CopyToDevice";
  if (dst_device == device_) {
    return InvalidArgument(
        "CopyToDevice cannot accept the same source and destination devices");
  }

  TF_ASSIGN_OR_RETURN(
      LocalDeviceState * dst_local_device,
      tensorflow::down_cast<PjRtStreamExecutorDevice*>(dst_device)
          ->GetLocalDeviceState());
  LocalDeviceState* transfer_local_device = device_->local_device_state();

  CHECK_EQ(dst_local_device->allocation_model(),
           transfer_local_device->allocation_model());

  se::Stream* transfer_stream =
      transfer_local_device->GetDeviceToDeviceStream();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ITEXPjRtBuffer> dst_buffer,
      AllocateITEXDestinationBuffer(device_ordinal_, device_, client_,
                                    dimensions_, element_type_, buffer_size()));

  transfer_stream->ThenMemcpyD2D(&dst_buffer->buffer(), buffer(),
                                 dst_buffer->buffer_size());
  if (!transfer_stream->ok()) {
    return InternalError("Device->Device Memcpy failed.");
  }

  return std::move(dst_buffer);
}

StatusOr<size_t> ITEXPjRtBuffer::GetOnDeviceSizeInBytes() const {
  return buffer_.size();
}

void ITEXPjRtBuffer::Delete() {
  VLOG(1) << "ITEXPjRtBuffer::Delete";
  Status status = allocator_->Deallocate(device_ordinal_, buffer_);
  if (!status.ok()) {
    LOG(ERROR) << "Buffer deallocation failed: " << status;
  }
}

bool ITEXPjRtBuffer::IsOnCpu() const {
  return client()->platform_id() == CpuId();
}

PjRtFuture<Status> ITEXPjRtBuffer::GetReadyFuture() {
  auto promise = PjRtFuture<Status>::CreatePromise();
  promise.Set(OkStatus());
  return PjRtFuture<Status>(std::move(promise));
}

PjRtFuture<Status> ITEXPjRtBuffer::ToLiteral(MutableLiteralBase* literal) {
  LocalDeviceState* local_device = device_->local_device_state();
  se::Stream* stream = local_device->GetDeviceToHostStream();

  auto promise = PjRtFuture<Status>::CreatePromise();
  se::DeviceMemoryBase base = buffer();

  auto* executor = stream->parent()->implementation();
  tsl::Status status =
      executor->SynchronousMemcpy(literal->untyped_data(), base, buffer_size());

  promise.Set(status);
  return PjRtFuture<Status>(std::move(promise));
}

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
ITEXPjRtBuffer::AcquireExternalReference() {
  return std::make_unique<ITEXBufferExternalReference>(buffer().opaque());
}

PjRtFuture<Status> ITEXPjRtBuffer::CopyRawToHost(void* dst, int64_t offset,
                                                 int64_t transfer_size) {
  return PjRtFuture<Status>(
      Unimplemented("Raw copies to host not implemented."));
}

void ITEXPjRtBuffer::CopyToRemoteDevice(
    PjRtFuture<StatusOr<std::string>> serialized_descriptor,
    RemoteSendCallback on_done) {
  DCHECK(false) << "CopyToRemoteDevice not implemented.";
}

void ITEXPjRtBuffer::CopyToRemoteDeviceScattered(
    PjRtFuture<StatusOr<std::vector<std::string>>> serialized_descriptors,
    std::vector<RemoteSendCallback> callbacks,
    const ScatterDetails& scatter_details) {
  DCHECK(false) << "CopyToRemoteDeviceScattered not implemented.";
}

}  // namespace xla
