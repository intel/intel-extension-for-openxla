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
#ifndef XLA_SERVICE_GPU_CCL_OPS_H_
#define XLA_SERVICE_GPU_CCL_OPS_H_
#include <vector>

#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/ccl_collective_thunk.h"
#include "xla/stream_executor/sycl/sycl_types.h"

#if !ITEX_USE_CCL

namespace xla {
namespace gpu {

void sycl_allreduce(const void* send_buffer, void* recv_buffer,
                    int element_count, PrimitiveType dtype,
                    ReductionKind reduction_kind,
                    se::gpu::GpuStreamHandle gpu_stream, ncclComm_t comm);

void sycl_allgather(const void* send_buffer, void* recv_buffer,
                    int element_count, PrimitiveType dtype,
                    se::gpu::GpuStreamHandle gpu_stream, ncclComm_t comm);

void sycl_alltoall(std::vector<const void*> send_buffer,
                   std::vector<void*> recv_buffer, int element_count,
                   PrimitiveType dtype, se::gpu::GpuStreamHandle gpu_stream,
                   ncclComm_t comm);

void sycl_reduce_scatter(const void* send_buffer, void* recv_buffer,
                         int element_count, PrimitiveType dtype,
                         ReductionKind reduction_kind,
                         se::gpu::GpuStreamHandle gpu_stream, ncclComm_t comm);

void sycl_collective_permute(const void* send_buffer, void* recv_buffer,
                             int element_count, PrimitiveType dtype,
                             const std::optional<int64_t>& source_id,
                             const std::optional<int64_t>& target_id,
                             se::gpu::GpuStreamHandle gpu_stream,
                             ncclComm_t comm);
}  // namespace gpu
}  // namespace xla

#endif  // ITEX_USE_CCL
#endif  // XLA_SERVICE_GPU_CCL_OPS_H_
