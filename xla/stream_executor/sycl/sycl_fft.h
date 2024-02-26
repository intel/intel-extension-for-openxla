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

#ifndef XLA_STREAM_EXECUTOR_SYCL_FFT_H_
#define XLA_STREAM_EXECUTOR_SYCL_FFT_H_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include "oneapi/mkl/blas.hpp"
#include "oneapi/mkl/dfti.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/lapack.hpp"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {

class Stream;

namespace gpu {

class GpuExecutor;

class SYCLPlan_Helper {
 public:
  SYCLPlan_Helper()
      : is_initialized_(false),
        scale_factor_(-1.0),
        dims_vec_(std::vector<int64_t> (1, 0)),
        batch_count_(-1),
        mkl_istrides_(std::vector<int64_t> (1, 0)),
        mkl_ostrides_(std::vector<int64_t> (1, 0)),
        tmp_istride_(-1),
        tmp_ostride_(-1),
        input_distance_(0),
        output_distance_(0),
        is_forward_(false),
        is_real_(false) {}

  ~SYCLPlan_Helper() { }

  tsl::Status Initialize(double scale_factor, std::vector<int64_t>& dims_vec,
                         int batch_count, std::vector<int64_t>& mkl_istrides,
                         std::vector<int64_t>& mkl_ostrides,
                         int64_t tmp_istride, int64_t tmp_ostride,
                         uint64 input_distance, uint64 output_distance,
                         bool is_forward, bool is_real) {
    if (IsInitialized()) {
      return tsl::errors::Internal("SYCLPlan_Helper is already initialized.");
    }
    scale_factor_ = scale_factor;
    dims_vec_ = dims_vec;
    batch_count_ = batch_count;
    mkl_istrides_ = mkl_istrides;
    mkl_ostrides_ = mkl_ostrides;
    tmp_istride_ = tmp_istride;
    tmp_ostride_ = tmp_ostride;
    input_distance_ = input_distance;
    output_distance_ = output_distance;
    is_initialized_ = true;
    is_forward_ = is_forward;
    is_real_ = is_real;
    return ::tsl::OkStatus();
  }

  bool IsInitialized() const { return is_initialized_; }
  double get_scale_factor() const { return scale_factor_; }
  int get_batch_count() const { return batch_count_; }
  const std::vector<int64_t> &get_dims_vec() const { return dims_vec_; }
  const std::vector<int64_t> &get_mkl_istrides() const { return mkl_istrides_; }
  const std::vector<int64_t> &get_mkl_ostrides() const { return mkl_ostrides_; }
  int64_t get_tmp_istride() const { return tmp_istride_; }
  int64_t get_tmp_ostride() const { return tmp_ostride_; }
  uint64 get_input_distance() const { return input_distance_; }
  uint64 get_output_distance() const { return output_distance_; }
  bool get_is_forward() const { return is_forward_; }
  bool get_is_real() const { return is_real_; }

 private:
  bool is_initialized_;
  double scale_factor_;
  std::vector<std::int64_t> dims_vec_;
  int batch_count_;
  std::vector<int64_t> mkl_istrides_;
  std::vector<int64_t> mkl_ostrides_;
  int64_t tmp_istride_;
  int64_t tmp_ostride_;
  uint64 input_distance_;
  uint64 output_distance_;
  bool is_forward_;
  bool is_real_;
};

class SYCLFftPlan : public fft::Plan {
 public:
  SYCLFftPlan()
      : parent_(nullptr),
        plan_(nullptr),
        fft_type_(fft::Type::kInvalid),
        scratch_(nullptr),
        scratch_size_bytes_(0),
        is_initialized_(false),
        scratch_allocator_(nullptr) {}
  ~SYCLFftPlan() override;

  const std::unique_ptr<SYCLPlan_Helper> &GetPlan(){
    if (IsInitialized()) {
      return plan_;
    } else {
      LOG(FATAL) << "Try to get syfftPlan before initialization.";
    }
  }

  // Initialize function for batched plan
  tsl::Status Initialize(GpuExecutor* parent, Stream* stream, int rank,
                         uint64_t* elem_count, uint64_t* input_embed,
                         uint64_t input_stride, uint64 input_distance,
                         uint64_t* output_embed, uint64_t output_stride,
                         uint64_t output_distance, fft::Type type,
                         int batch_count, ScratchAllocator* scratch_allocator);

  // Initialize function for 1d,2d, and 3d plan
  tsl::Status Initialize(GpuExecutor* parent, Stream* stream, int rank,
                         uint64_t* elem_count, fft::Type type,
                         ScratchAllocator* scratch_allocator);

  tsl::Status UpdateScratchAllocator(Stream* stream,
                                     ScratchAllocator* scratch_allocator);

  ScratchAllocator* GetScratchAllocator() const { return scratch_allocator_; }

 protected:
  bool IsInitialized() const { return is_initialized_; }

 private:
  GpuExecutor* parent_;
  std::unique_ptr<SYCLPlan_Helper> plan_;
  fft::Type fft_type_;
  DeviceMemory<uint8_t> scratch_;
  size_t scratch_size_bytes_;
  bool is_initialized_;
  ScratchAllocator* scratch_allocator_;
};

class SYCLFft : public fft::FftSupport {
 public:
  explicit SYCLFft(GpuExecutor* parent) : parent_(parent) {}
  ~SYCLFft() override {}

  TENSORFLOW_STREAM_EXECUTOR_GPU_FFT_SUPPORT_OVERRIDES

 private:
  GpuExecutor* parent_;

  SYCLFft(const SYCLFft&) = delete;
  void operator=(const SYCLFft&) = delete;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_FFT_H_
