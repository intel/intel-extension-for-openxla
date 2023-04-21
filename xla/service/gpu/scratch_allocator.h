#ifndef XLA_SERVICE_GPU_SCRATCH_ALLOCATOR_H_
#define XLA_SERVICE_GPU_SCRATCH_ALLOCATOR_H_
#include "xla/stream_executor/scratch_allocator.h"

namespace xla {
namespace gpu {
tsl::Status AllocateWorkspace(
    void** workspace, stream_executor::ScratchAllocator* scratch_allocator,
    size_t num_bytes);

}  // namespace gpu
}  // namespace xla
#endif  // XLA_SERVICE_GPU_SCRATCH_ALLOCATOR_H_