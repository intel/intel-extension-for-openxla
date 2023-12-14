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

#ifndef XLA_SERVICE_GPU_REDUNDANT_BITCAST_MOVER_H_
#define XLA_SERVICE_GPU_REDUNDANT_BITCAST_MOVER_H_

#include <functional>
#include <utility>

#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

class RedundantBitcastMover : public HloModulePass {
 public:
  RedundantBitcastMover() = default;

  absl::string_view name() const override { return "redundant-bitcast-mover"; }
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_REDUNDANT_BITCAST_MOVER_H_
