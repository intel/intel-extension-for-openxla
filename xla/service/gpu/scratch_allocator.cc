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

#include "xla/service/gpu/scratch_allocator.h"

namespace xla {
namespace gpu {
tsl::Status AllocateWorkspace(
    void** workspace, stream_executor::ScratchAllocator* scratch_allocator,
    size_t num_bytes) {
  TF_ASSIGN_OR_RETURN(stream_executor::DeviceMemory<tsl::uint8> workspace_bytes,
                      scratch_allocator->AllocateBytes(num_bytes));
  *workspace = static_cast<void*>(workspace_bytes.opaque());
  return tsl::OkStatus();
}

}  // namespace gpu
}  // namespace xla