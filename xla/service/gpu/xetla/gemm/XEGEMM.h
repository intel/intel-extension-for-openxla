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
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/sycl/hw_info.h"
#include "xla/stream_executor/sycl/sycl_stream.h"

using namespace gpu::xetla;
namespace se = ::stream_executor;

class HGEMM_XETLA final {
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
  void* epilogue_tensors_[MAX_EPILOGUES];
  EpilogueType epilogue_types_[MAX_EPILOGUES];
  float epilogue_params_[MAX_EPILOGUES];
  int num_epilogues_ = 0;
  bool is_a_row_major_;
  bool is_a_col_major_;
  bool is_b_row_major_;
  bool is_b_col_major_;
  bool fallback_;
  int m_, n_, k_;
  int selected_policy_;
  float alpha_ = 1.0f;

 public:
  HGEMM_XETLA() = default;
  bool fallback() const { return fallback_; }
  HGEMM_XETLA& add_alpha(const float alpha) {
    alpha_ = alpha;
    return *this;
  }
  HGEMM_XETLA& add_matrix_c(const xla::gpu::MatrixDescriptor& c) {
    c_ = const_cast<xla::gpu::MatrixDescriptor*>(&c);
    return *this;
  }
  HGEMM_XETLA& add_matrix_a(const xla::gpu::MatrixDescriptor& a) {
    a_ = const_cast<xla::gpu::MatrixDescriptor*>(&a);
    return *this;
  }
  HGEMM_XETLA& add_matrix_b(const xla::gpu::MatrixDescriptor& b) {
    b_ = const_cast<xla::gpu::MatrixDescriptor*>(&b);
    return *this;
  }

  HGEMM_XETLA& add_epilogue(const void* t, EpilogueType eptype,
                            const float x = 1.0) {
    epilogue_tensors_[num_epilogues_] = const_cast<void*>(t);
    epilogue_params_[num_epilogues_] = x;
    epilogue_types_[num_epilogues_++] = eptype;
    return *this;
  }

  HGEMM_XETLA& build() {
    fallback_ = true;
    is_a_row_major_ = (a_->transpose == se::blas::Transpose::kNoTranspose);
    is_a_col_major_ = (a_->transpose == se::blas::Transpose::kTranspose);
    is_b_row_major_ = (b_->transpose == se::blas::Transpose::kNoTranspose);
    is_b_col_major_ = (b_->transpose == se::blas::Transpose::kTranspose);
    m_ = is_a_row_major_ ? a_->num_rows : a_->num_cols;
    k_ = is_a_row_major_ ? a_->num_cols : a_->num_rows;
    n_ = is_b_row_major_ ? b_->num_cols : b_->num_rows;
    if (is_a_col_major_) return *this;
    fallback_ = false;
    selected_policy_ = select_gemm_config(m_, n_, k_, is_b_row_major_,
                                          64);  // 64 is subslice count per tile
    return *this;
  }

  void run(se::gpu::GpuStreamHandle handle) {
    using scalar_t = sycl::half;
    sycl::queue q = *handle;
    if (num_epilogues_ == 0) {
      CHECK(alpha_ == 1.0f);
      hgemm_common_policies[selected_policy_](
          q, reinterpret_cast<sycl::half*>(c_->data.opaque()),
          reinterpret_cast<sycl::half*>(a_->data.opaque()),
          reinterpret_cast<sycl::half*>(b_->data.opaque()), m_, n_, k_);
    } else if (num_epilogues_ == 1 && epilogue_types_[0] == RES_ADD) {
      if (alpha_ == 1.0f) {
        hgemm_res_policies[selected_policy_](
            q, reinterpret_cast<sycl::half*>(c_->data.opaque()),
            reinterpret_cast<sycl::half*>(a_->data.opaque()),
            reinterpret_cast<sycl::half*>(b_->data.opaque()),
            reinterpret_cast<sycl::half*>(epilogue_tensors_[0]), m_, n_, k_,
            epilogue_params_[0]);
      } else {
        hgemm_addmm_policies[selected_policy_](
            q, reinterpret_cast<sycl::half*>(c_->data.opaque()),
            reinterpret_cast<sycl::half*>(epilogue_tensors_[0]),
            reinterpret_cast<sycl::half*>(a_->data.opaque()),
            reinterpret_cast<sycl::half*>(b_->data.opaque()), m_, n_, k_,
            alpha_, epilogue_params_[0]);
      }
    } else if (num_epilogues_ == 2 && epilogue_types_[0] == RES_ADD &&
               epilogue_types_[1] == RES_ADD) {
      CHECK(alpha_ == 1.0f);
      hgemm_res_res_policies[selected_policy_](
          q, reinterpret_cast<sycl::half*>(c_->data.opaque()),
          reinterpret_cast<sycl::half*>(a_->data.opaque()),
          reinterpret_cast<sycl::half*>(b_->data.opaque()),
          reinterpret_cast<sycl::half*>(epilogue_tensors_[0]),
          reinterpret_cast<sycl::half*>(epilogue_tensors_[1]), m_, n_, k_,
          epilogue_params_[0], epilogue_params_[1]);
    } else if (num_epilogues_ == 1 && epilogue_types_[0] == BIAS) {
      CHECK(alpha_ == 1.0f);
      hgemm_bias_policies[selected_policy_](
          q, reinterpret_cast<sycl::half*>(c_->data.opaque()),
          reinterpret_cast<sycl::half*>(a_->data.opaque()),
          reinterpret_cast<sycl::half*>(b_->data.opaque()),
          reinterpret_cast<sycl::half*>(epilogue_tensors_[0]), m_, n_, k_,
          epilogue_params_[0]);
    } else if (num_epilogues_ == 2 && epilogue_types_[0] == BIAS &&
               epilogue_types_[1] == SCALED_RES_ADD) {
      CHECK(alpha_ == 1.0f);
      hgemm_bias_res_policies[selected_policy_](
          q, reinterpret_cast<sycl::half*>(c_->data.opaque()),
          reinterpret_cast<sycl::half*>(a_->data.opaque()),
          reinterpret_cast<sycl::half*>(b_->data.opaque()),
          reinterpret_cast<sycl::half*>(epilogue_tensors_[0]),
          reinterpret_cast<sycl::half*>(epilogue_tensors_[1]), m_, n_, k_,
          epilogue_params_[0], epilogue_params_[1]);
    } else if (num_epilogues_ == 3 && epilogue_types_[0] == BIAS &&
               epilogue_types_[1] == RES_ADD && epilogue_types_[2] == RES_ADD) {
      CHECK(alpha_ == 1.0f);
      hgemm_bias_res_res_policies[selected_policy_](
          q, reinterpret_cast<sycl::half*>(c_->data.opaque()),
          reinterpret_cast<sycl::half*>(a_->data.opaque()),
          reinterpret_cast<sycl::half*>(b_->data.opaque()),
          reinterpret_cast<sycl::half*>(epilogue_tensors_[0]),
          reinterpret_cast<sycl::half*>(epilogue_tensors_[1]),
          reinterpret_cast<sycl::half*>(epilogue_tensors_[2]), m_, n_, k_,
          epilogue_params_[0], epilogue_params_[1], epilogue_params_[2]);
    } else if (num_epilogues_ == 2 && epilogue_types_[0] == BIAS &&
               epilogue_types_[1] == GELU) {
      CHECK(alpha_ == 1.0f);
      hgemm_bias_gelu_policies[selected_policy_](
          q, reinterpret_cast<sycl::half*>(c_->data.opaque()),
          reinterpret_cast<sycl::half*>(a_->data.opaque()),
          reinterpret_cast<sycl::half*>(b_->data.opaque()),
          reinterpret_cast<sycl::half*>(epilogue_tensors_[0]), m_, n_, k_,
          epilogue_params_[0]);
    } else if (num_epilogues_ == 1 && epilogue_types_[0] == RES_MUL) {
      CHECK(alpha_ == 1.0f);
      hgemm_resmul_policies[selected_policy_](
          q, reinterpret_cast<sycl::half*>(c_->data.opaque()),
          reinterpret_cast<sycl::half*>(a_->data.opaque()),
          reinterpret_cast<sycl::half*>(b_->data.opaque()),
          reinterpret_cast<sycl::half*>(epilogue_tensors_[0]), m_, n_, k_);
    } else if (num_epilogues_ == 1 && epilogue_types_[0] == SILU) {
      CHECK(alpha_ == 1.0f);
      hgemm_silu_policies[selected_policy_](
          q, reinterpret_cast<sycl::half*>(c_->data.opaque()),
          reinterpret_cast<sycl::half*>(a_->data.opaque()),
          reinterpret_cast<sycl::half*>(b_->data.opaque()), m_, n_, k_);
    } else {
      LOG(ERROR) << "No mateched policy";
    }
  }
};
