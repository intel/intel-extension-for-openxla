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

#ifndef XLA_PJRT_SE_XPU_PJRT_CLIENT_H_
#define XLA_PJRT_SE_XPU_PJRT_CLIENT_H_
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/statusor.h"

namespace stream_executor {

class MultiDeviceAdapter;

}

namespace xla {

class StreamExecutorXpuDevice : public PjRtStreamExecutorDevice {
 public:
  StreamExecutorXpuDevice(int id,
                          std::unique_ptr<LocalDeviceState> local_device_state,
                          std::string device_kind, std::string device_vendor,
                          int node_id, int slice_index = 0);

  int slice_index() const;
  absl::string_view device_vendor();

 private:
  std::string device_vendor_;
  int slice_index_;
};

StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorXpuClient(
    bool asynchronous, const GpuAllocatorConfig& allocator_config, int node_id,
    int num_nodes = 1,
    const std::optional<std::set<int>>& allowed_devices = std::nullopt,
    std::optional<std::string> platform_name = std::nullopt,
    bool should_stage_host_to_device_transfers = true,
    PjRtClient::KeyValueGetCallback kv_get = nullptr,
    PjRtClient::KeyValuePutCallback kv_put = nullptr,
    bool enable_mock_nccl = false);

}  // namespace xla

#endif  // XLA_PJRT_SE_XPU_PJRT_CLIENT_H_
