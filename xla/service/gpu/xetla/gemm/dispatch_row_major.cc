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

#include "xla/service/gpu/xetla/gemm/dispatch_row_major.h"

#include "xla/service/gpu/xetla/gemm/gemm_common.h"
#include "xla/service/gpu/xetla/gemm/gemm_dispatch.h"
#include "xla/stream_executor/gpu/gpu_types.h"

namespace gpu {
namespace xetla {

template <typename ComputeType>
bool GemmRowMajorDispatcher<ComputeType>::run(se::gpu::GpuStreamHandle handle) {
  int WG_M = std::get<0>(selected_policy_id_);
  int WG_N = std::get<1>(selected_policy_id_);
  int SG_M = std::get<2>(selected_policy_id_);
  int SG_N = std::get<3>(selected_policy_id_);
  int SG_K = std::get<4>(selected_policy_id_);
  int SLM_KS = std::get<5>(selected_policy_id_);
  return gemm_policy<ComputeType>::call(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,
                                        this, handle);
}

template <typename ComputeType>
template <int WG_M, int WG_N, int SG_M, int SG_N, int SG_K, int SLM_KS>
bool GemmRowMajorDispatcher<ComputeType>::dispatch(
    se::gpu::GpuStreamHandle handle) {
  return do_dispatch<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, true>(
      handle, params_);
}

template class GemmRowMajorDispatcher<sycl::half>;
template class GemmRowMajorDispatcher<gpu::xetla::bf16>;

}  // namespace xetla
}  // namespace gpu
