/* Copyright (c) 2023 Intel Corporation

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

#pragma once

#include "xla/service/gpu/matrix_descriptor.h"
#include "xla/service/gpu/xetla/gemm/gemm.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/sycl/hw_info.h"
#include "xla/stream_executor/sycl/sycl_stream.h"
#include "xla/stream_executor/sycl/sycl_types.h"

using namespace gpu::xetla;
namespace se = ::stream_executor;

#define HGEMM_DISPATCH(F)                                            \
  {                                                                  \
    F(q, reinterpret_cast<sycl::half*>(c_->data.opaque()),           \
      reinterpret_cast<sycl::half*>(a_->data.opaque()),              \
      reinterpret_cast<sycl::half*>(b_->data.opaque()), m_, n_, k_); \
  }

#define HGEMM_BIAS_DISPATCH(F)                                   \
  {                                                              \
    F(q, reinterpret_cast<sycl::half*>(c_->data.opaque()),       \
      reinterpret_cast<sycl::half*>(a_->data.opaque()),          \
      reinterpret_cast<sycl::half*>(b_->data.opaque()),          \
      reinterpret_cast<sycl::half*>(epilogues_[0]), m_, n_, k_); \
  }

#define HGEMM_BIAS_RES_RES_DISPATCH(F)                           \
  {                                                              \
    F(q, reinterpret_cast<sycl::half*>(c_->data.opaque()),       \
      reinterpret_cast<sycl::half*>(a_->data.opaque()),          \
      reinterpret_cast<sycl::half*>(b_->data.opaque()),          \
      reinterpret_cast<sycl::half*>(epilogues_[0]),              \
      reinterpret_cast<sycl::half*>(epilogues_[1]),              \
      reinterpret_cast<sycl::half*>(epilogues_[2]), m_, n_, k_); \
  }

#define HGEMM_BIAS_GELU_DISPATCH(F)                              \
  {                                                              \
    F(q, reinterpret_cast<sycl::half*>(c_->data.opaque()),       \
      reinterpret_cast<sycl::half*>(a_->data.opaque()),          \
      reinterpret_cast<sycl::half*>(b_->data.opaque()),          \
      reinterpret_cast<sycl::half*>(epilogues_[0]), m_, n_, k_); \
  }

#define HGEMM_RESMUL_DISPATCH(F)                                 \
  {                                                              \
    F(q, reinterpret_cast<sycl::half*>(c_->data.opaque()),       \
      reinterpret_cast<sycl::half*>(a_->data.opaque()),          \
      reinterpret_cast<sycl::half*>(b_->data.opaque()),          \
      reinterpret_cast<sycl::half*>(epilogues_[0]), m_, n_, k_); \
  }

#define HGEMM_SILU_DISPATCH(F)                                       \
  {                                                                  \
    F(q, reinterpret_cast<sycl::half*>(c_->data.opaque()),           \
      reinterpret_cast<sycl::half*>(a_->data.opaque()),              \
      reinterpret_cast<sycl::half*>(b_->data.opaque()), m_, n_, k_); \
  }

#define HGEMM_RES_DISPATCH(F)                                    \
  {                                                              \
    F(q, reinterpret_cast<sycl::half*>(c_->data.opaque()),       \
      reinterpret_cast<sycl::half*>(a_->data.opaque()),          \
      reinterpret_cast<sycl::half*>(b_->data.opaque()),          \
      reinterpret_cast<sycl::half*>(epilogues_[0]), m_, n_, k_); \
  }

#define HGEMM_BIAS_XRES_DISPATCH(F)                                            \
  {                                                                            \
    F(q, reinterpret_cast<sycl::half*>(c_->data.opaque()),                     \
      reinterpret_cast<sycl::half*>(a_->data.opaque()),                        \
      reinterpret_cast<sycl::half*>(b_->data.opaque()),                        \
      reinterpret_cast<sycl::half*>(epilogues_[0]),                            \
      reinterpret_cast<sycl::half*>(epilogues_[1]), (scalar_t)pf32[1], m_, n_, \
      k_);                                                                     \
  }

#define HGEMM_COMMON_DISPATCH_IMPL(DISPATCHER, F) \
  if (is_b_row_major_)                            \
    DISPATCHER(F##true_)                          \
  else                                            \
    DISPATCHER(F##false_)

#define HGEMM_COMMON_DISPATCH(F)                                               \
  {                                                                            \
    if (num_epilogues_ == 0)                                                   \
      HGEMM_COMMON_DISPATCH_IMPL(HGEMM_DISPATCH, hgemm##F)                     \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == BIAS)                 \
      HGEMM_COMMON_DISPATCH_IMPL(HGEMM_BIAS_DISPATCH, hgemm_bias##F)           \
    else if (num_epilogues_ == 3 && epilogue_type_[0] == BIAS &&               \
             epilogue_type_[1] == RES_ADD && epilogue_type_[2] == RES_ADD)     \
      HGEMM_COMMON_DISPATCH_IMPL(HGEMM_BIAS_RES_RES_DISPATCH,                  \
                                 hgemm_bias_res_res##F)                        \
    else if (num_epilogues_ == 2 && epilogue_type_[0] == BIAS &&               \
             epilogue_type_[1] == GELU)                                        \
      HGEMM_COMMON_DISPATCH_IMPL(HGEMM_BIAS_GELU_DISPATCH, hgemm_bias_gelu##F) \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == RES_MUL)              \
      HGEMM_COMMON_DISPATCH_IMPL(HGEMM_RESMUL_DISPATCH, hgemm_resmul##F)       \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == SILU)                 \
      HGEMM_COMMON_DISPATCH_IMPL(HGEMM_SILU_DISPATCH, hgemm_silu##F)           \
    else if (num_epilogues_ == 1 && epilogue_type_[0] == RES_ADD)              \
      HGEMM_COMMON_DISPATCH_IMPL(HGEMM_RES_DISPATCH, hgemm_res##F)             \
    else if (num_epilogues_ == 2 && epilogue_type_[0] == BIAS &&               \
             epilogue_type_[1] == SCALED_RES_ADD)                              \
      HGEMM_COMMON_DISPATCH_IMPL(HGEMM_BIAS_XRES_DISPATCH, hgemm_bias_res##F)  \
    else                                                                       \
      LOG(FATAL) << "Unsupported DISPATCHER";                                  \
  }

class HGEMMXetla final {
 public:
  enum EpilogueType {
    BIAS = 0,
    RES_ADD,
    GELU,
    RES_MUL,
    SILU,
    SCALED_RES_ADD,
  };

 private:
  enum {
    MAX_EPILOGUES = 4,
  };
  xla::gpu::MatrixDescriptor *a_, *b_, *c_;
  void* epilogues_[MAX_EPILOGUES];
  EpilogueType epilogue_type_[MAX_EPILOGUES];
  float pf32[MAX_EPILOGUES];
  int num_epilogues_ = 0;
  bool is_a_row_major_;
  bool is_a_col_major_;
  bool is_b_row_major_;
  bool is_b_col_major_;
  bool fallback_;
  int m_, n_, k_;

 public:
  HGEMMXetla() = default;
  bool fallback() const { return fallback_; }
  HGEMMXetla& add_matrix_c(const xla::gpu::MatrixDescriptor& c) {
    c_ = const_cast<xla::gpu::MatrixDescriptor*>(&c);
    return *this;
  }
  HGEMMXetla& add_matrix_a(const xla::gpu::MatrixDescriptor& a) {
    a_ = const_cast<xla::gpu::MatrixDescriptor*>(&a);
    return *this;
  }
  HGEMMXetla& add_matrix_b(const xla::gpu::MatrixDescriptor& b) {
    b_ = const_cast<xla::gpu::MatrixDescriptor*>(&b);
    return *this;
  }
  HGEMMXetla& add_epilogue(void* t, EpilogueType eptype) {
    epilogues_[num_epilogues_] = const_cast<void*>(t);
    epilogue_type_[num_epilogues_++] = eptype;
    return *this;
  }
  HGEMMXetla& add_epilogue(const void* t, EpilogueType eptype, const float x) {
    epilogues_[num_epilogues_] = const_cast<void*>(t);
    pf32[num_epilogues_] = x;
    epilogue_type_[num_epilogues_++] = eptype;
    return *this;
  }

  HGEMMXetla& build() {
    fallback_ = true;

    is_a_row_major_ = (a_->transpose == se::blas::Transpose::kNoTranspose);
    is_a_col_major_ = (a_->transpose == se::blas::Transpose::kTranspose);
    is_b_row_major_ = (b_->transpose == se::blas::Transpose::kNoTranspose);
    is_b_col_major_ = (b_->transpose == se::blas::Transpose::kTranspose);
    m_ = is_a_row_major_ ? a_->num_rows : a_->num_cols;
    k_ = is_a_row_major_ ? a_->num_cols : a_->num_rows;
    n_ = is_b_row_major_ ? b_->num_cols : b_->num_rows;
    if (is_a_col_major_) return *this;
    if (!(n_ >= 4096 && k_ >= 1024)) return *this;
    fallback_ = false;
    return *this;
  }

  void run(se::gpu::GpuStreamHandle handle) {
    using scalar_t = sycl::half;
    sycl::queue q = *handle;
    if (m_ == 60 && n_ == 4096 && k_ == 4096) {
      HGEMM_COMMON_DISPATCH(_32x64_8x16x32_2_);
    } else if (m_ == 60 && (n_ >= 16384) && k_ == 4096) {
      HGEMM_COMMON_DISPATCH(_256x256_32x64x16_1_);
    } else if (m_ >= 1024) {
      HGEMM_COMMON_DISPATCH(_256x256_32x64x32_1_);
    } else if (m_ >= 32) {
      HGEMM_COMMON_DISPATCH(_32x256_8x32x16_1_);
    } else if (n_ == 13824 && (k_ == 4096 || k_ == 5120)) {
      HGEMM_COMMON_DISPATCH(
          _8x512_8x32x16_2_);  // HGEMM_IMPL_FUNC(8, 256, 8, 32, 16, 2, false)
      return;
    } else if ((n_ == 4096 || n_ == 5120) && k_ == 13824) {
      HGEMM_COMMON_DISPATCH(_8x128_8x16x16_4_);
      return;
    } else if (n_ >= 4096 && n_ < 5120) {
      HGEMM_COMMON_DISPATCH(_32x64_8x16x16_2_);
      return;
    } else if (n_ >= 5120 && n_ < 11008) {
      HGEMM_COMMON_DISPATCH(_8x128_8x16x16_4_);  // 8, 128, 8, 16, 16, 4
      return;
    } else if (n_ >= 11008 && n_ < 13824) {
      HGEMM_COMMON_DISPATCH(_16x256_8x16x16_1_);  // 16, 256, 8, 16, 16, 1
      return;
    } else {
      HGEMM_COMMON_DISPATCH(_8x512_8x16x16_1_);  // 8, 512, 8, 16, 16, 1
      return;
    }
  }
};
