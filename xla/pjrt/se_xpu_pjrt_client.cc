/* Copyright (c) 2023 Intel Corporation

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

#include "xla/pjrt/se_xpu_pjrt_client.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "xla/client/client_library.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/platform_util.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/integrations/device_host_allocator.h"
#include "xla/stream_executor/integrations/device_mem_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"

namespace xla {
namespace {

class StreamExecutorXpuClient : public xla::PjRtStreamExecutorClient {
 public:
  using xla::PjRtStreamExecutorClient::PjRtStreamExecutorClient;

  xla::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;
};

xla::StatusOr<xla::DeviceAssignment>
StreamExecutorXpuClient::GetDefaultDeviceAssignment(int num_replicas,
                                                    int num_partitions) const {
  if (num_partitions == 1 && num_replicas <= addressable_devices().size()) {
    xla::DeviceAssignment assignment(num_replicas, 1);
    for (int i = 0; i < num_replicas; ++i) {
      assignment(i, 0) = addressable_devices().at(i)->id();
    }
    return assignment;
  }
  // Fallback to default global device assignment if we can't run locally.
  return PjRtStreamExecutorClient::GetDefaultDeviceAssignment(num_replicas,
                                                              num_partitions);
}

// Builds a LocalDeviceState for each GPU present.
StatusOr<std::map<int, std::unique_ptr<LocalDeviceState>>>
BuildLocalDeviceStates(LocalClient* xla_client) {
  std::map<int, std::unique_ptr<LocalDeviceState>> addressable_devices;
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    addressable_devices.emplace(
        executor->device_ordinal(),
        std::make_unique<LocalDeviceState>(
            executor, xla_client, LocalDeviceState::kComputeSynchronized,
            /*max_inflight_computations=*/32,
            /*allow_event_reuse=*/true, /*use_callback_stream=*/true));
  }
  return std::move(addressable_devices);
}

// Constructs a GPU device memory allocator to use, according to the allocator
// configuration the client requested.
StatusOr<std::unique_ptr<se::DeviceMemoryAllocator>>
GetStreamExecutorXpuDeviceAllocator(
    se::Platform* platform, const GpuAllocatorConfig& allocator_config,
    const std::map<int, std::unique_ptr<LocalDeviceState>>&
        addressable_devices) {
  std::unique_ptr<se::DeviceMemoryAllocator> allocator;
  switch (allocator_config.kind) {
    case GpuAllocatorConfig::Kind::kCudaAsync: {
      LOG(ERROR) << "Async allocator is not supported; falling back to BFC.";
      [[fallthrough]];
    }

    case GpuAllocatorConfig::Kind::kDefault:
    case GpuAllocatorConfig::Kind::kBFC: {
      LOG(INFO) << "Using BFC allocator.";
      std::vector<se::StreamExecutor*> executors;
      executors.reserve(addressable_devices.size());
      std::vector<se::MultiDeviceAdapter::AllocatorWithStream>
          allocators_and_streams;
      for (const auto& ordinal_and_device : addressable_devices) {
        TF_ASSIGN_OR_RETURN(
            auto bfc_allocator,
            CreateBFCAllocator(ordinal_and_device.second->executor(),
                               allocator_config.memory_fraction,
                               allocator_config.preallocate));
        allocators_and_streams.emplace_back(
            std::move(bfc_allocator),
            ordinal_and_device.second->compute_stream());
      }
      allocator = std::make_unique<se::MultiDeviceAdapter>(
          platform, std::move(allocators_and_streams));
      break;
    }

    case GpuAllocatorConfig::Kind::kPlatform:
      LOG(INFO) << "Using platform allocator.";
      break;
  }
  return std::move(allocator);
}

std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  for (auto& ordinal_and_device : local_device_states) {
    const se::DeviceDescription& description =
        ordinal_and_device.second->executor()->GetDeviceDescription();
    auto device = std::make_unique<StreamExecutorXpuDevice>(
        ordinal_and_device.first, std::move(ordinal_and_device.second),
        description.name(), description.device_vendor(), node_id);
    devices.push_back(std::move(device));
  }
  return devices;
}

inline const char* XpuName() {
  static constexpr char kXpuName[] = "xpu";
  return kXpuName;
}

inline PjRtPlatformId XpuId() {
  static const PjRtPlatformId kXpuId = tsl::Fingerprint64(XpuName());
  return kXpuId;
}

}  // namespace

StreamExecutorXpuDevice::StreamExecutorXpuDevice(
    int id, std::unique_ptr<LocalDeviceState> local_device_state,
    std::string device_kind, std::string device_vendor, int node_id,
    int slice_index)
    : PjRtStreamExecutorDevice(id, std::move(local_device_state),
                               std::move(device_kind), node_id),
      device_vendor_(std::move(device_vendor)),
      slice_index_(slice_index) {
  description().SetAttributes({
      {"device_vendor", std::string("Intel")},
      {"slice_index", static_cast<int64_t>(slice_index)},
  });
  description().SetToString(
      absl::StrFormat("IntelXpuDevice(id=%i, process_index=%i, slice_index=%i)",
                      id, process_index(), slice_index));
}

int StreamExecutorXpuDevice::slice_index() const { return slice_index_; }

absl::string_view StreamExecutorXpuDevice::device_vendor() {
  return device_vendor_;
}

StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorXpuClient(
    bool asynchronous, const GpuAllocatorConfig& allocator_config, int node_id,
    int num_nodes, const std::optional<std::set<int>>& allowed_devices,
    std::optional<std::string> platform_name,
    bool should_stage_host_to_device_transfers,
    PjRtClient::KeyValueGetCallback kv_get,
    PjRtClient::KeyValuePutCallback kv_put, bool enable_mock_nccl) {
  TF_ASSIGN_OR_RETURN(LocalClient * xla_client,
                      GetGpuXlaClient(platform_name, allowed_devices));
  std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states;
  TF_ASSIGN_OR_RETURN(local_device_states, BuildLocalDeviceStates(xla_client));
  // EnablePeerAccess(xla_client->backend().stream_executors());
  TF_ASSIGN_OR_RETURN(
      // SYCL: hardcode to static variable due to a bug for sycl alloc api.
      static std::unique_ptr<se::DeviceMemoryAllocator> allocator,
      GetStreamExecutorXpuDeviceAllocator(
          xla_client->platform(), allocator_config, local_device_states));
  auto host_memory_allocator =
      GetGpuHostAllocator(local_device_states.begin()->second->executor());

  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  auto gpu_run_options = std::make_unique<gpu::GpuExecutableRunOptions>();
  if (num_nodes > 1) {
    TF_RET_CHECK(kv_get != nullptr);
    TF_RET_CHECK(kv_put != nullptr);
    // TF_RETURN_IF_ERROR(BuildDistributedDevices(
    //     std::move(local_device_states), node_id, num_nodes, &devices,
    //     gpu_run_options.get(), kv_get, kv_put));
    return Unimplemented("BuildDistributedDevices not");
  } else {
    devices = BuildLocalDevices(std::move(local_device_states), node_id);
  }
  return std::unique_ptr<PjRtClient>(std::make_unique<StreamExecutorXpuClient>(
      XpuName(), xla_client, std::move(devices),
      /*node_id=*/node_id, std::move(allocator),
      std::move(host_memory_allocator), should_stage_host_to_device_transfers,
      /*gpu_run_options=*/std::move(gpu_run_options)));
}

}  // namespace xla
