/* Copyright (c) 2024 Intel Corporation

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

#ifndef XLA_SERVICE_GPU_GEMM_ALGORITHM_PICKER_H_
#define XLA_SERVICE_GPU_GEMM_ALGORITHM_PICKER_H_

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// GemmAlgorithmPicker supports two modes: device and deviceless.
// In device mode, we run autotuning on the device and store autotune results.
// In deviceless mode, we pass in some information related to the device and
// use stored autotune results to rewrite Gemm instructions. If the required
// autotune result is not stored, then algorithm is set to kRuntimeAutotuning.
class GemmAlgorithmPicker : public HloModulePass {
 public:
  explicit GemmAlgorithmPicker(AutotuneConfig config) : config_(config) {}

  absl::string_view name() const override { return "gemm-impl-picker"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  AutotuneConfig config_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GEMM_ALGORITHM_PICKER_H_
