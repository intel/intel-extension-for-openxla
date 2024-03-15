/* Copyright (c) 2024 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_STREAM_EXECUTOR_SYCL_BLAS_H_
#define XLA_STREAM_EXECUTOR_SYCL_BLAS_H_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "oneapi/mkl/blas.hpp"
#include "oneapi/mkl/dfti.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/lapack.hpp"
#include "xla/stream_executor/blas.h"

namespace stream_executor {

class Stream;

namespace gpu {
class GpuExecutor;
}  // namespace gpu

using syclStream_t = ::sycl::queue *;

namespace sycl {
// Thread-safe post-initialization.
class SYCLBlas : public blas::BlasSupport {
 public:
  explicit SYCLBlas(gpu::GpuExecutor *parent);

  bool Init();

  ~SYCLBlas() override;

  TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES

  gpu::BlasLt *GetBlasLt() override { return nullptr; }

 private:
  bool SetStream(Stream *stream) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  syclStream_t SYCLStream(Stream *stream);

  absl::Mutex mu_;

  gpu::GpuExecutor *parent_;

  void *blas_ ABSL_GUARDED_BY(mu_);

  void *blas_it_;

  SYCLBlas(const SYCLBlas &) = delete;
  void operator=(const SYCLBlas &) = delete;
};

}  // namespace sycl
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_BLAS_H_
