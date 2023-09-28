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
#ifndef XLA_SERVICE_GPU_XETLA_GEMM_H_
#define XLA_SERVICE_GPU_XETLA_GEMM_H_
#include <sycl/sycl.hpp>

#include "absl/strings/str_cat.h"
#include "xla/service/gpu/matrix_descriptor.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/sycl/sycl_stream.h"

namespace se = ::stream_executor;

namespace gpu {
namespace xetla {

extern std::tuple<int, int, int, int, int, int> selectXetlaGemmConfig(int m,
                                                                      int n,
                                                                      int k);

extern std::tuple<int, int, int, int, int, int> selectXetlaQKVGemmConfig(int m,
                                                                         int n,
                                                                         int k);

template <typename ComputeType>
class XetlaGemmKernel {
 public:
  enum EpilogueType {
    BIAS = 0,
    RES_ADD,
    GELU,
    RES_MUL,
    SILU,
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
  std::tuple<int, int, int, int, int, int> selected_policy_id_;
  float alpha_ = 1.0f;

 public:
  XetlaGemmKernel() = default;
  bool fallback() const { return fallback_; }
  XetlaGemmKernel& add_alpha(const float alpha) {
    alpha_ = alpha;
    return *this;
  }
  XetlaGemmKernel& add_matrix_c(const xla::gpu::MatrixDescriptor& c) {
    c_ = const_cast<xla::gpu::MatrixDescriptor*>(&c);
    return *this;
  }
  XetlaGemmKernel& add_matrix_a(const xla::gpu::MatrixDescriptor& a) {
    a_ = const_cast<xla::gpu::MatrixDescriptor*>(&a);
    return *this;
  }
  XetlaGemmKernel& add_matrix_b(const xla::gpu::MatrixDescriptor& b) {
    b_ = const_cast<xla::gpu::MatrixDescriptor*>(&b);
    return *this;
  }
  XetlaGemmKernel& add_epilogue(const void* t, EpilogueType eptype,
                                const float x = 1.0) {
    epilogue_tensors_[num_epilogues_] = const_cast<void*>(t);
    epilogue_params_[num_epilogues_] = x;
    epilogue_types_[num_epilogues_++] = eptype;
    return *this;
  }
  XetlaGemmKernel& build() {
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
    selected_policy_id_ = selectXetlaGemmConfig(m_, n_, k_);
    return *this;
  }

  template <int WG_M, int WG_N, int SG_M, int SG_N, int SG_K, int SLM_KS>
  void dispatch(se::gpu::GpuStreamHandle handle);

  void run(se::gpu::GpuStreamHandle handle);
};

template <typename ComputeType>
class XetlaQKVGemmKernel : public XetlaGemmKernel<ComputeType> {
 private:
  xla::gpu::MatrixDescriptor *a_, *b_;
  xla::gpu::MatrixDescriptor* q_out_;
  xla::gpu::MatrixDescriptor* k_out_;
  xla::gpu::MatrixDescriptor* v_out_;
  bool is_a_row_major_;
  bool is_a_col_major_;
  bool is_b_row_major_;
  bool is_b_col_major_;
  bool fallback_;
  int m_, n_, k_;
  std::tuple<int, int, int, int, int, int> selected_policy_id_;
  float alpha_ = 1.0f;

 public:
  XetlaQKVGemmKernel() = default;
  XetlaQKVGemmKernel& add_matrix_q_out(
      const xla::gpu::MatrixDescriptor& q_out) {
    q_out_ = const_cast<xla::gpu::MatrixDescriptor*>(&q_out);
    return *this;
  }
  XetlaQKVGemmKernel& add_matrix_k_out(
      const xla::gpu::MatrixDescriptor& k_out) {
    k_out_ = const_cast<xla::gpu::MatrixDescriptor*>(&k_out);
    return *this;
  }
  XetlaQKVGemmKernel& add_matrix_v_out(
      const xla::gpu::MatrixDescriptor& v_out) {
    v_out_ = const_cast<xla::gpu::MatrixDescriptor*>(&v_out);
    return *this;
  }
  XetlaQKVGemmKernel& add_matrix_a(const xla::gpu::MatrixDescriptor& a) {
    a_ = const_cast<xla::gpu::MatrixDescriptor*>(&a);
    return *this;
  }
  XetlaQKVGemmKernel& add_matrix_b(const xla::gpu::MatrixDescriptor& b) {
    b_ = const_cast<xla::gpu::MatrixDescriptor*>(&b);
    return *this;
  }
  XetlaQKVGemmKernel& build() {
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
    selected_policy_id_ = selectXetlaQKVGemmConfig(m_, n_, k_);
    return *this;
  }

  template <int WG_M, int WG_N, int SG_M, int SG_N, int SG_K, int SLM_KS>
  void dispatch(se::gpu::GpuStreamHandle handle);

  void run(se::gpu::GpuStreamHandle handle);
};

}  // namespace xetla
}  // namespace gpu

#endif  // XLA_SERVICE_GPU_XETLA_GEMM_H_
