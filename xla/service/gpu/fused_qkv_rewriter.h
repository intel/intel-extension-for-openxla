/* Copyright (c) 2023 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_SERVICE_GPU_FUSED_QKV_REWRITER_H_
#define XLA_SERVICE_GPU_FUSED_QKV_REWRITER_H_

#include <optional>

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// 0) Before QKV fusion   1) After QKV fusion
//
//          p                    p
//          |                    |
//          v                    v
//          I                    I
//        / | \                  |
//       |  |  |           +--fusion---+
//       v  v  v           |   / | \   |
//       Q  K  V           |  Q  K  V  |
//       \  |  /           |  |  |  |  |
//        v v v            |  v  v  v  |
//        ROOT             |   tuple   |
//                         +-----------+
//                          /    |    \           
//                        gte_1  |   gte_2
//                         |     |     |
//                         \     |    /
//                           v   v   v
//                             ROOT
//
class FusedQKVRewriter : public HloModulePass {
 public:
  explicit FusedQKVRewriter(const se::DeviceDescription& d,
                            se::CudaComputeCapability cuda_compute_capability)
      : device_info_(d), cuda_compute_capability_(cuda_compute_capability) {}

  absl::string_view name() const override { return "xelta-qkv-rewriter"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::CudaComputeCapability cuda_compute_capability_;

  // The reachability map of current computation.
  std::unique_ptr<HloReachabilityMap> reachability_;

  const se::DeviceDescription device_info_;

  bool FuseQKVSiblings(HloComputation* computation, HloInstruction* parent,
                       FusionInfoCache* fusion_info_cache);

  StatusOr<bool> DoQKVMultiOutputFusion(HloComputation* computation);

  // Recompute reachability for the current computation.
  void RecomputeReachability(HloComputation* computation);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSED_QKV_REWRITER_H_
