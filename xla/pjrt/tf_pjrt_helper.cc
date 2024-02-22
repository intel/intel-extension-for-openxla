#include "xla/pjrt/tf_pjrt_helper.h"

#include "xla/pjrt/itex_pjrt_buffer.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/tf_xpu_pjrt_client.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"
#include "xla/stream_executor/sycl/sycl_stream.h"
#include "xla/stream_executor/tf_allocator_adapter.h"
#include "xla/stream_executor/tpu/c_api_conversions.h"

xla::PrimitiveType XlaDataTypeFromString(std::string data_type) {
  if (data_type == "bool")
    return xla::PRED;
  else if (data_type == "int8" || data_type == "qint8")
    return xla::S8;
  else if (data_type == "int16" || data_type == "qint16")
    return xla::S16;
  else if (data_type == "int32" || data_type == "qint32")
    return xla::S32;
  else if (data_type == "int64")
    return xla::S64;
  else if (data_type == "uint8" || data_type == "quint8")
    return xla::U8;
  else if (data_type == "uint16" || data_type == "quint16")
    return xla::U16;
  else if (data_type == "uint32")
    return xla::U32;
  else if (data_type == "uint64")
    return xla::U64;
  else if (data_type == "bfloat16")
    return xla::BF16;
  else if (data_type == "half")
    return xla::F16;
  else if (data_type == "float")
    return xla::F32;
  else if (data_type == "double")
    return xla::F64;
  else if (data_type == "complex64")
    return xla::C64;
  else if (data_type == "complex128")
    return xla::C128;
  else
    return xla::PRIMITIVE_TYPE_INVALID;
}

void* ITEXOpaqueDataPointerFromPjRtBuffer(PJRT_Buffer* pjrt_c_buffer) {
  std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference_hold;
  external_reference_hold =
      std::move(pjrt_c_buffer->buffer->AcquireExternalReference().value());
  return external_reference_hold->OpaqueDeviceMemoryDataPointer();
}

PJRT_Buffer* ITEXCreateSEPjRtBuffer(int device_id, std::string data_type,
                                    std::vector<int64_t> dimentions,
                                    std::vector<int64_t> layout,
                                    PJRT_Client* pjrt_c_client) {
  xla::PjRtDevice* pjrt_device =
      pjrt_c_client->client->LookupDevice(device_id).value();
  xla::PrimitiveType type = XlaDataTypeFromString(data_type);
  xla::Shape shape =
      xla::ShapeUtil::MakeShapeWithDenseLayout(type, dimentions, layout);
  auto* pjrt_buffer = new PJRT_Buffer{
      std::move(
          pjrt_c_client->client->CreateUninitializedBuffer(shape, pjrt_device)
              .value()),
      pjrt_c_client};
  auto* pjrt_se_buffer = dynamic_cast<xla::PjRtStreamExecutorBuffer*>(pjrt_buffer->buffer.get());
  pjrt_se_buffer->set_allocate_by_third_party_framework();
  return pjrt_buffer;
}

PJRT_Buffer* ITEXCreatePjRtBuffer(int device_id, std::string data_type,
                                  std::vector<int64_t>* dimentions, size_t size,
                                  PJRT_Client* pjrt_c_client) {
  xla::PjRtDevice* pjrt_device =
      pjrt_c_client->client->LookupDevice(device_id).value();
  xla::PrimitiveType type = XlaDataTypeFromString(data_type);
  std::unique_ptr<xla::ITEXPjRtBuffer> buffer =
      AllocateITEXDestinationBuffer(device_id, pjrt_device,
                                    pjrt_c_client->client.get(), *dimentions,
                                    type, size)
          .value();
  return new PJRT_Buffer{std::move(buffer), pjrt_c_client};
}

void ITEXDeletePjRtBuffer(PJRT_Buffer* pjrt_buffer) {
  //printf("CBOSS I am in ITEXDeletePjRtBuffer!\r\n");
  //fflush(stdout);
  delete pjrt_buffer;
}

void ITEXRecoverPjRtBuffer(PJRT_Buffer* pjrt_buffer) {
  auto* buffer = reinterpret_cast<xla::PjRtStreamExecutorBuffer*>(pjrt_buffer->buffer.get());
  buffer->recover_buffer();
}


void* ITEXGetStreamFromPjRtDevice(int device_id, PJRT_Client* pjrt_c_client) {
  xla::PjRtDevice* pjrt_device =
      std::move(pjrt_c_client->client->LookupDevice(device_id).value());
  void* stream = static_cast<void*>(stream_executor::gpu::AsGpuStreamValue(
      (static_cast<xla::PjRtStreamExecutorDevice*>(pjrt_device))
          ->local_device_state()
          ->compute_stream()));
  return stream;
}

void* ITEXBFCAllocateOnSyclDevice(const sycl::device& device,
                                  PJRT_Client* pjrt_c_client, size_t n) {
  int device_id = SYCLGetDeviceOrdinal(device, &device_id);
  auto pjrt_client = reinterpret_cast<xla::PjRtStreamExecutorClient*>(
      pjrt_c_client->client.get());
  auto* allocator = reinterpret_cast<stream_executor::MultiDeviceAdapter*>(
      pjrt_client->allocator());
  void* device_mem = allocator->AllocateRaw(device_id, n, true, 0);
  return device_mem;
}

void ITEXBFCDeallocateOnSyclDevice(const sycl::device& device,
                                   PJRT_Client* pjrt_c_client, void* addr) {
  int device_id = SYCLGetDeviceOrdinal(device, &device_id);
  auto pjrt_client = reinterpret_cast<xla::PjRtStreamExecutorClient*>(
      pjrt_c_client->client.get());
  auto* allocator = reinterpret_cast<stream_executor::MultiDeviceAdapter*>(
      pjrt_client->allocator());
  allocator->DeallocateRaw(device_id, addr);
}

extern "C" {
struct PjRtBuffer_Info {
  size_t size;
  std::string datatype;
  std::vector<int64_t> dimensions;
  std::vector<int64_t> layout;
};

void* C_ITEXOpaqueDataPointerFromPjRtBuffer(PJRT_Buffer* pjrt_c_buffer) {
  return ITEXOpaqueDataPointerFromPjRtBuffer(pjrt_c_buffer);
}

PJRT_Buffer* C_ITEXCreatePjRtBuffer(int device_id,
                                    PjRtBuffer_Info* pjrt_buffer_info,
                                    PJRT_Client* pjrt_c_client) {
  return ITEXCreatePjRtBuffer(device_id, pjrt_buffer_info->datatype,
                              &(pjrt_buffer_info->dimensions),
                              pjrt_buffer_info->size, pjrt_c_client);
}

PJRT_Buffer* C_ITEXCreateSEPjRtBuffer(int device_id,
                                      PjRtBuffer_Info* pjrt_buffer_info,
                                      PJRT_Client* pjrt_c_client) {
  return ITEXCreateSEPjRtBuffer(device_id, pjrt_buffer_info->datatype,
                                pjrt_buffer_info->dimensions,
                                pjrt_buffer_info->layout, pjrt_c_client);
}

void* C_ITEXGetStreamFromPjRtDevice(int device_id, PJRT_Client* pjrt_c_client) {
  return ITEXGetStreamFromPjRtDevice(device_id, pjrt_c_client);
}

void C_RegisterCustomCallTarget(const char* symbol, void* address,
                                const char* platform) {
  xla::CustomCallTargetRegistry::Global()->Register(symbol, address, platform);
}
}

namespace xla {

// Ensures that it is safe to deallocate any buffers that have been enqueued in
// an operation on stream. Called only in rare error cases that are triggered
// during enqueue. These cases generally correspond to resource exhaustion.
void StallStreamOnError(LocalDeviceState* local_device, se::Stream* stream) {
  switch (local_device->allocation_model()) {
    case LocalDeviceState::kAsynchronous:
      // We can safely deallocate any dangling buffers immediately. NOTE: this
      // assumes that any buffers enqueued on stream are local to stream's
      // executor, and manual action may be needed if that condition is not met.
      break;

    case LocalDeviceState::kComputeSynchronized:
      // This will stall computation but that's ok in this very rare error
      // case.
      if (stream != local_device->compute_stream()) {
        local_device->compute_stream()->ThenWaitFor(stream);
      }
      break;

    case LocalDeviceState::kSynchronous:
      // This will stall the calling thread but that's ok in this very rare
      // error case. If the stall fails just crash, since we have no other
      // way to synchronize.
      TF_CHECK_OK(stream->BlockHostUntilDone());
      break;
  }
}

// Adds necessary synchronization after a copy has been enqueued to a buffer.
// definition_event was added when the buffer was allocated, but has not yet
// had an event recorded.
Status AddDestinationBufferSynchronization(
    LocalDeviceState* local_device,
    std::shared_ptr<BufferSequencingEvent> definition_event,
    se::Stream* copy_stream) {
  StatusOr<EventPool::Handle> event_or =
      local_device->event_pool().ThenAllocateAndRecordEvent(copy_stream);
  if (!event_or.ok()) {
    StallStreamOnError(local_device, copy_stream);
    return event_or.status();
  }
  definition_event->SetSequencingEvent(std::move(event_or).value(),
                                       copy_stream);
  return OkStatus();
}

PJRT_Buffer* SameDevicePjRtBufferCopy(PJRT_Buffer* src_buffer,
                                      PJRT_Client* c_client) {
  PjRtStreamExecutorBuffer* se_src_buffer =
      static_cast<PjRtStreamExecutorBuffer*>(src_buffer->buffer.get());
  if (se_src_buffer->on_device_shape().IsTuple()) {
    std::cout << "ITEXSameDevicePjRtBufferCopy does not support Tupple yet"
              << std::endl;
    std::abort();
  }

  PjRtStreamExecutorDevice* pjrt_device = se_src_buffer->device();
  LocalDeviceState* transfer_local_device = pjrt_device->local_device_state();

  se::Stream* transfer_stream =
      transfer_local_device->GetDeviceToDeviceStream();

  auto* se_client =
      static_cast<PjRtStreamExecutorClient*>(c_client->client.get());
  TransferManager* transfer_manager =
      se_client->client()->backend().transfer_manager();

  ScopedShapedBuffer dst_buffer =
      transfer_manager
          ->AllocateScopedShapedBuffer(se_src_buffer->on_device_shape(),
                                       se_client->allocator(),
                                       transfer_local_device->device_ordinal())
          .value();
  transfer_stream->ThenWaitFor(transfer_local_device->compute_stream());

  absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 2>
      definition_events;
  definition_events.emplace_back(
      std::make_shared<BufferSequencingEvent>(se_client->thread_pool()));

  std::shared_ptr<TrackedDeviceBuffer> dst_device_buffer =
      TrackedDeviceBuffer::FromScopedShapedBuffer(&dst_buffer,
                                                  definition_events);
  auto py_dst_buffer = std::make_unique<PjRtStreamExecutorBuffer>(
      se_src_buffer->on_device_shape(), std::move(dst_device_buffer), se_client,
      pjrt_device);

  ShapedBuffer shaped_dst_buffer = py_dst_buffer->AsShapedBuffer().value();

  PjRtStreamExecutorBuffer::ScopedHold scoped_dst_buffer(
      py_dst_buffer->GetBufferWithUsageHold());
  // Copy the leaf buffers.
  ShapedBuffer shaped_src_buffer = se_src_buffer->AsShapedBuffer().value();

  StatusOr<std::shared_ptr<BufferSequencingEvent>> copy_event_or =
      [&]() -> StatusOr<std::shared_ptr<BufferSequencingEvent>> {
    for (const auto& leaf : shaped_src_buffer.buffers().leaves()) {
      const ShapeIndex& index = leaf.first;
      const se::DeviceMemoryBase& input_buffer = leaf.second;
      const se::DeviceMemoryBase& output_buffer =
          shaped_dst_buffer.buffer(index);
      TF_RET_CHECK(input_buffer.size() == output_buffer.size())
          << "input: " << input_buffer.size()
          << " output: " << output_buffer.size();
      if (input_buffer.size() != 0) {
        TF_RETURN_IF_ERROR(transfer_local_device->ThenMemcpyDeviceToDevice(
            transfer_stream, transfer_local_device->compute_stream(),
            input_buffer, output_buffer));
      }
    }
    std::shared_ptr<BufferSequencingEvent> event =
        scoped_dst_buffer->definition_events()[0];
    TF_RETURN_IF_ERROR(AddDestinationBufferSynchronization(
        transfer_local_device, event, transfer_stream));
    return event;
  }();

  if (!copy_event_or.ok()) {
    StallStreamOnError(transfer_local_device, transfer_stream);
  }

  return new PJRT_Buffer{std::move(py_dst_buffer), c_client};
}

PJRT_Buffer* SameDeviceITEXBufferCopy(PJRT_Buffer* src_buffer,
                                      PJRT_Client* c_client) {
  ITEXPjRtBuffer* itex_src_buffer =
      static_cast<ITEXPjRtBuffer*>(src_buffer->buffer.get());
  PjRtStreamExecutorDevice* pjrt_device = itex_src_buffer->device();

  LocalDeviceState* transfer_local_device = pjrt_device->local_device_state();
  auto* se_client =
      static_cast<PjRtStreamExecutorClient*>(c_client->client.get());

  se::Stream* transfer_stream =
      transfer_local_device->GetDeviceToDeviceStream();
  std::unique_ptr<ITEXPjRtBuffer> dst_buffer =
      AllocateITEXDestinationBuffer(
          transfer_local_device->device_ordinal(), pjrt_device, se_client,
          itex_src_buffer->dimensions(), itex_src_buffer->element_type(),
          itex_src_buffer->buffer_size())
          .value();

  transfer_stream->ThenMemcpyD2D(&dst_buffer->buffer(),
                                 itex_src_buffer->buffer(),
                                 dst_buffer->buffer_size());
  if (!transfer_stream->ok()) {
    LOG(FATAL) << "NextPluggableDevice->NextPluggableDevice Memcpy "
               << "failed.";
  }

  return new PJRT_Buffer{std::move(dst_buffer), c_client};
}

}  // namespace xla

PJRT_Buffer* ITEXSameDevicePjRtBufferCopy(PJRT_Buffer* src_buffer,
                                          PJRT_Client* c_client,
                                          bool xla_enabled) {
  if (xla_enabled) {
    return xla::SameDevicePjRtBufferCopy(src_buffer, c_client);
  } else {
    return xla::SameDeviceITEXBufferCopy(src_buffer, c_client);
  }
}

void ITEXXlaShapeToDeviceShapeRepresentation(void* serialized_xla_shape,
                                             void* serialized_device_shape) {
  xla::Shape xla_shape =
      ApiConverter::FromC(static_cast<XLA_Shape*>(serialized_xla_shape));
  ApiConverter::ToC(xla_shape,
                    static_cast<XLA_Shape*>(serialized_device_shape));
}
