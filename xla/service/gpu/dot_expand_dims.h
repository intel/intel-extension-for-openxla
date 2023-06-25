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

#ifndef ITEX_CORE_COMPILER_XLA_SERVICE_DOT_EXPAND_DIMS_H_
#define ITEX_CORE_COMPILER_XLA_SERVICE_DOT_EXPAND_DIMS_H_

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Expand dot dims for dimension 1 so that it can call onednn.
class DotExpandDims : public HloModulePass {
 public:
  explicit DotExpandDims();
  absl::string_view name() const override { return "dot-expand-dims"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // ITEX_CORE_COMPILER_XLA_SERVICE_DOT_EXPAND_DIMS_H_