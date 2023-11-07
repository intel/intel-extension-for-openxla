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

#include <memory>
#include <utility>

#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/se_xpu_pjrt_client.h"

namespace pjrt {
namespace xpu_plugin {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  // TODO(b/261916900) initializing allocator_config is important as should be
  // passed through the args later.
  xla::GpuAllocatorConfig allocator_config;
  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtClient> client,
      xla::GetStreamExecutorXpuClient(
          /*asynchronous=*/false, allocator_config,
          /*node_id=*/0, /*num_nodes*/ 1,
          /*allowed_devices*/ std::nullopt, /*platform_name*/ "SYCL"));
  args->client = pjrt::CreateWrapperClient(std::move(client));
  return nullptr;
}

PJRT_Error* PJRT_XpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{tsl::errors::Unimplemented(
      "Topology not supported for XPU compilation.")};
}

constexpr PJRT_Api pjrt_api =
    pjrt::CreatePjrtApi(pjrt::xpu_plugin::PJRT_Client_Create,
                        pjrt::xpu_plugin::PJRT_XpuDeviceTopology_Create);

const PJRT_Api* GetXpuPjrtApi() { return &pjrt_api; }

}  // namespace xpu_plugin
}  // namespace pjrt
