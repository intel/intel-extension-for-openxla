/* Copyright (c) 2023 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/computation_placer.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

static std::unique_ptr<xla::ComputationPlacer> CreateSYCLComputationPlacer() {
  return std::make_unique<xla::ComputationPlacer>();
}

static bool InitModule() {
  xla::ComputationPlacer::RegisterComputationPlacer(
      stream_executor::sycl::kSyclPlatformId, &CreateSYCLComputationPlacer);
  return true;
}
static bool module_initialized = InitModule();