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

#ifndef XLA_SERVICE_GPU_MATRIX_DESCRIPTOR_H_
#define XLA_SERVICE_GPU_MATRIX_DESCRIPTOR_H_

#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

// This struct contains the metadata of a matrix, e.g., its base address and
// dimensions.
struct MatrixDescriptor {
  se::DeviceMemoryBase data;
  se::blas::Transpose transpose;
  int64_t num_rows;
  int64_t num_cols;
  int64_t batch_stride;
  int64_t leading_dim_stride;

  int64_t reduced_dim() const {
    return transpose == se::blas::Transpose::kTranspose ? num_rows : num_cols;
  }

  template <typename T>
  se::DeviceMemory<T> cast() const {
    return se::DeviceMemory<T>(data);
  }
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MATRIX_DESCRIPTOR_H_
