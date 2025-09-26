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

// CUDA userspace driver library wrapper functionality.

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_DRIVER_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_DRIVER_H_

#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

namespace stream_executor {
namespace gpu {

class GpuContext {
 public:
  GpuContext(sycl::device* d, sycl::context* c) : device_(d), context_(c) {}

  sycl::device* device() const { return device_; }
  sycl::context* context() const { return context_; }

  // Disallow copying and moving.
  GpuContext(GpuContext&&) = delete;
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

 private:
  sycl::device* device_;
  sycl::context* context_;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_DRIVER_H_