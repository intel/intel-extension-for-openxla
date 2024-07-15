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

#ifndef XLA_SERVICE_GPU_XETLA_GEMM_DISPATCH_COL_MAJOR_H_
#define XLA_SERVICE_GPU_XETLA_GEMM_DISPATCH_COL_MAJOR_H_

#include "xla/service/gpu/xetla/gemm/gemm_common.h"
#include "xla/service/gpu/xetla/gemm/gemm_dispatch.h"
#include "xla/stream_executor/gpu/gpu_types.h"

namespace gpu {
namespace xetla {

template <typename ComputeType>
class GemmColMajorDispatcher {
 public:
  GemmColMajorDispatcher() = default;

  GemmColMajorDispatcher(
      DispatchParams* params,
      std::tuple<int, int, int, int, int, int> selected_policy_id)
      : params_(params), selected_policy_id_(selected_policy_id) {}

  template <int WG_M, int WG_N, int SG_M, int SG_N, int SG_K, int SLM_KS>
  bool dispatch(se::gpu::GpuStreamHandle handle);

  bool run(se::gpu::GpuStreamHandle handle);

 private:
  DispatchParams* params_;
  std::tuple<int, int, int, int, int, int> selected_policy_id_;
};

}  // namespace xetla
}  // namespace gpu

#endif  // XLA_SERVICE_GPU_XETLA_GEMM_DISPATCH_COL_MAJOR_H_