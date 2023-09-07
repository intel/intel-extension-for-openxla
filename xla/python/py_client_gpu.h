/* Copyright (c) 2023 Intel Corporation

Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_PYTHON_PY_CLIENT_GPU_H_
#define XLA_PYTHON_PY_CLIENT_GPU_H_

#include "xla/service/custom_call_status.h"
#include "xla/stream_executor/gpu/gpu_types.h"

using gpuStreamHandle = stream_executor::gpu::GpuStreamHandle;

namespace xla {

void XlaPythonGpuCallback(gpuStreamHandle stream, void** buffers,
                          const char* opaque, size_t opaque_len,
                          XlaCustomCallStatus* status);

}  // namespace xla

#endif  // XLA_PYTHON_PY_CLIENT_GPU_H_
