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

// Defines the GpuStream type - the CUDA-specific implementation of the generic
// StreamExecutor Stream interface.

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_STREAM_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_STREAM_H_

#include "xla/stream_executor/gpu/gpu_stream.h"

namespace stream_executor {
namespace sycl {

using SYCLStream = gpu::GpuStream;

inline SYCLStream* AsSYCLStream(Stream* stream) {
  return gpu::AsGpuStream(stream);
}

}  // namespace sycl
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_STREAM_H_
