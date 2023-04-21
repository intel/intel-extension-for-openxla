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