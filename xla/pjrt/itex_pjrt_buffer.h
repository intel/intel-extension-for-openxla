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

#ifndef ITEX_CORE_COMPILER_XLA_PJRT_ITEX_PJRT_BUFFER_H_
#define ITEX_CORE_COMPILER_XLA_PJRT_ITEX_PJRT_BUFFER_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/tf_allocator_adapter.h"

namespace xla {

class ITEXBufferExternalReference : public PjRtBuffer::ExternalReference {
 public:
  ITEXBufferExternalReference(void* data_ptr) { data_ptr_ = data_ptr; }
  ~ITEXBufferExternalReference() override {}
};

class ITEXPjRtBuffer : public PjRtBuffer {
 public:
  ITEXPjRtBuffer::ITEXPjRtBuffer(int device_id,
                                 se::DeviceMemoryBase device_memory,
                                 absl::Span<const int64_t> dimensions,
                                 PrimitiveType element_type, PjRtClient* client,
                                 PjRtDevice* device);

  ~ITEXPjRtBuffer() override;

  se::DeviceMemoryBase& buffer() { return buffer_; }
  const se::DeviceMemoryBase& buffer() const { return buffer_; }
  size_t buffer_size() const { return buffer_.size(); }
  PrimitiveType element_type() const override { return element_type_; }
  absl::Span<const int64_t> dimensions() const { return dimensions_; }
  PjRtStreamExecutorDevice* device() const override { return device_; }
  PjRtStreamExecutorClient* client() const override { return client_; }
  PjRtMemorySpace* memory_space() const override { return nullptr; }
  const Shape& on_device_shape() const override { return shape_; }

  StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device) override;

  inline void Delete() override;

  inline bool IsDeleted() override { return false; }

  PjRtFuture<Status> ToLiteral(MutableLiteralBase* literal) override;

  bool IsOnCpu() const override;

  PjRtFuture<Status> CopyRawToHost(void* dst, int64_t offset,
                                   int64_t transfer_size) override;

  void CopyToRemoteDevice(
      PjRtFuture<StatusOr<std::string>> serialized_descriptor,
      RemoteSendCallback on_done) override;

  void CopyToRemoteDeviceScattered(
      PjRtFuture<StatusOr<std::vector<std::string>>> serialized_descriptors,
      std::vector<RemoteSendCallback> callbacks,
      const ScatterDetails& scatter_details) override;

  PjRtFuture<Status> GetReadyFuture() override;

  StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override {
    return Unimplemented("Implement CopyToMemorySpace");
  }

  StatusOr<std::unique_ptr<ExternalReference>> ReleaseDeviceMemoryOwnership(
      bool wait_for_operations_to_complete) override {
    return Unimplemented("Implement ReleaseDeviceMemoryOwnership");
  };

 inline void record_memory_allocation_size(size_t size);
 inline size_t get_recorded_memory_allocation_size();
 inline void set_hold_by_third_party_framework(bool value);
 inline void set_hold_by_framework(bool value);
 inline bool recover_buffer();
 inline bool is_hold_by_third_party_framework();
 inline bool is_hold_by_framework();

 private:
  int device_ordinal_;
  bool isHoldByThirdPartyFramwork_ = false;
  bool isHoldByFramwork_ = true;
  bool need_bfc_deallocate_ = true;
  size_t MemoryAllocationByteSize_ = 0;
  Shape shape_;
  absl::InlinedVector<int64_t, 6> dimensions_;
  PrimitiveType element_type_;
  stream_executor::MultiDeviceAdapter* allocator_;
  PjRtStreamExecutorDevice* device_;
  PjRtStreamExecutorClient* client_;
  se::DeviceMemoryBase buffer_;
};

StatusOr<std::unique_ptr<ITEXPjRtBuffer>> AllocateITEXDestinationBuffer(
    int device_id, PjRtDevice* device, PjRtClient* client,
    absl::Span<const int64_t> dimensions, PrimitiveType element_type,
    size_t size);

}  // namespace xla

#endif  // ITEX_CORE_COMPILER_XLA_PJRT_ITEX_PJRT_BUFFER_H_
