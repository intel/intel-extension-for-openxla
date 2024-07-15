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

#ifndef XLA_SERVICE_GPU_XETLA_GEMM_GEMM_DISPATCH_H_
#define XLA_SERVICE_GPU_XETLA_GEMM_GEMM_DISPATCH_H_

#include "xla/service/gpu/matrix_descriptor.h"
#include "xla/service/gpu/xetla/gemm/gemm_common.h"
#include "xla/service/gpu/xetla/gemm/hgemm_impl.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/sycl/sycl_stream.h"

namespace se = ::stream_executor;

namespace gpu {
namespace xetla {

struct DispatchParams {
  xla::gpu::MatrixDescriptor *a_, *b_, *c_;
  int m_, n_, k_;
  float alpha_ = 1.0f;
  int num_epilogues_ = 0;
  void* epilogue_tensors_[kMaxNumEpilogues];
  EpilogueType epilogue_types_[kMaxNumEpilogues];
  float epilogue_params_[kMaxNumEpilogues];

  DispatchParams(xla::gpu::MatrixDescriptor* a, xla::gpu::MatrixDescriptor* b,
                 xla::gpu::MatrixDescriptor* c, int m, int n, int k,
                 float alpha, int num_epilogues, void* epilogue_tensors[],
                 EpilogueType epilogue_types[], float epilogue_params[])
      : a_(a),
        b_(b),
        c_(c),
        m_(m),
        n_(n),
        k_(k),
        alpha_(alpha),
        num_epilogues_(num_epilogues) {
    for (int i = 0; i < num_epilogues; i++) {
      epilogue_tensors_[i] = epilogue_tensors[i];
      epilogue_types_[i] = epilogue_types[i];
      epilogue_params_[i] = epilogue_params[i];
    }
  }
};

template <typename ComputeType, int WG_M, int WG_N, int SG_M, int SG_N,
          int SG_K, int SLM_KS, bool B_ROW_MAJOR>
bool do_dispatch(se::gpu::GpuStreamHandle handle, DispatchParams* params) {
  sycl::queue q = *handle;
  if (params->num_epilogues_ == 0) {
    hgemm_common<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
                 B_ROW_MAJOR>(
        q, reinterpret_cast<ComputeType*>(params->c_->data.opaque()),
        reinterpret_cast<ComputeType*>(params->a_->data.opaque()),
        reinterpret_cast<ComputeType*>(params->b_->data.opaque()), params->m_,
        params->n_, params->k_);
  } else if (params->num_epilogues_ == 1 &&
             params->epilogue_types_[0] == RES_ADD) {
    if (params->alpha_ == 1.0f) {
      hgemm_res<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
                B_ROW_MAJOR>(
          q, reinterpret_cast<ComputeType*>(params->c_->data.opaque()),
          reinterpret_cast<ComputeType*>(params->a_->data.opaque()),
          reinterpret_cast<ComputeType*>(params->b_->data.opaque()),
          reinterpret_cast<ComputeType*>(params->epilogue_tensors_[0]),
          params->m_, params->n_, params->k_, params->epilogue_params_[0]);
    } else {
      hgemm_addmm<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
                  B_ROW_MAJOR>(
          q, reinterpret_cast<ComputeType*>(params->c_->data.opaque()),
          reinterpret_cast<ComputeType*>(params->epilogue_tensors_[0]),
          reinterpret_cast<ComputeType*>(params->a_->data.opaque()),
          reinterpret_cast<ComputeType*>(params->b_->data.opaque()), params->m_,
          params->n_, params->k_, params->alpha_, params->epilogue_params_[0]);
    }
  } else if (params->num_epilogues_ == 1 &&
             params->epilogue_types_[0] == GELU) {
    CHECK(params->alpha_ == 1.0f);
    hgemm_gelu<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
               B_ROW_MAJOR>(
        q, reinterpret_cast<ComputeType*>(params->c_->data.opaque()),
        reinterpret_cast<ComputeType*>(params->a_->data.opaque()),
        reinterpret_cast<ComputeType*>(params->b_->data.opaque()), params->m_,
        params->n_, params->k_);
  } else if (params->num_epilogues_ == 1 &&
             params->epilogue_types_[0] == BIAS) {
    CHECK(params->alpha_ == 1.0f);
    hgemm_bias<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
               B_ROW_MAJOR>(
        q, reinterpret_cast<ComputeType*>(params->c_->data.opaque()),
        reinterpret_cast<ComputeType*>(params->a_->data.opaque()),
        reinterpret_cast<ComputeType*>(params->b_->data.opaque()),
        reinterpret_cast<ComputeType*>(params->epilogue_tensors_[0]),
        params->m_, params->n_, params->k_, params->epilogue_params_[0]);
  } else if (params->num_epilogues_ == 2 &&
             params->epilogue_types_[0] == BIAS &&
             params->epilogue_types_[1] == RES_ADD) {
    CHECK(params->alpha_ == 1.0f);
    hgemm_bias_res<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
                   B_ROW_MAJOR>(
        q, reinterpret_cast<ComputeType*>(params->c_->data.opaque()),
        reinterpret_cast<ComputeType*>(params->a_->data.opaque()),
        reinterpret_cast<ComputeType*>(params->b_->data.opaque()),
        reinterpret_cast<ComputeType*>(params->epilogue_tensors_[0]),
        reinterpret_cast<ComputeType*>(params->epilogue_tensors_[1]),
        params->m_, params->n_, params->k_, params->epilogue_params_[0],
        params->epilogue_params_[1]);
  } else if (params->num_epilogues_ == 2 &&
             params->epilogue_types_[0] == BIAS &&
             params->epilogue_types_[1] == GELU) {
    CHECK(params->alpha_ == 1.0f);
    hgemm_bias_gelu<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
                    B_ROW_MAJOR>(
        q, reinterpret_cast<ComputeType*>(params->c_->data.opaque()),
        reinterpret_cast<ComputeType*>(params->a_->data.opaque()),
        reinterpret_cast<ComputeType*>(params->b_->data.opaque()),
        reinterpret_cast<ComputeType*>(params->epilogue_tensors_[0]),
        params->m_, params->n_, params->k_, params->epilogue_params_[0]);

  } else {
    LOG(ERROR) << "No mateched policy, will fallback to oneDNN kernel";
    return false;
  }
  return true;
}

template <typename ComputeType, int WG_M, int WG_N, int SG_M, int SG_N,
          int SG_K, int SLM_KS>
struct GemmPolicy {
  template <typename DISPATCHER>
  static bool match_or_call(int wg_m, int wg_n, int sg_m, int sg_n, int sg_k,
                            int slm_ks, DISPATCHER* gemm_kernel,
                            se::gpu::GpuStreamHandle handle) {
    if (WG_M == wg_m && WG_N == wg_n && SG_M == sg_m && SG_N == sg_n &&
        SG_K == sg_k && SLM_KS == slm_ks) {
      return gemm_kernel
          ->template dispatch<WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS>(handle);
    }
    return false;
  }
};

template <typename ComputeType, typename MATCHER, typename... TArgs>
struct PolicyDispatcher {
  template <typename DISPATCHER>
  static bool call(int wg_m, int wg_n, int sg_m, int sg_n, int sg_k, int slm_ks,
                   DISPATCHER* gemm_kernel, se::gpu::GpuStreamHandle handle) {
    if (MATCHER::match_or_call(wg_m, wg_n, sg_m, sg_n, sg_k, slm_ks,
                               gemm_kernel, handle)) {
      return true;
    }
    return PolicyDispatcher<ComputeType, TArgs...>::call(
        wg_m, wg_n, sg_m, sg_n, sg_k, slm_ks, gemm_kernel, handle);
  }
};

template <typename ComputeType, typename MATCHER>
struct PolicyDispatcher<ComputeType, MATCHER> {
  template <typename DISPATCHER>
  static bool call(int wg_m, int wg_n, int sg_m, int sg_n, int sg_k, int slm_ks,
                   DISPATCHER* gemm_kernel, se::gpu::GpuStreamHandle handle) {
    if (MATCHER::match_or_call(wg_m, wg_n, sg_m, sg_n, sg_k, slm_ks,
                               gemm_kernel, handle)) {
      return true;
    }
    return false;
  }
};

template <typename ComputeType>
using gemm_policy =
    PolicyDispatcher<ComputeType, GemmPolicy<ComputeType, 8, 64, 8, 16, 32, 8>,
                     GemmPolicy<ComputeType, 8, 64, 8, 16, 16, 4>,
                     GemmPolicy<ComputeType, 8, 32, 8, 16, 16, 4>,
                     GemmPolicy<ComputeType, 8, 32, 8, 16, 16, 8>,
                     GemmPolicy<ComputeType, 8, 128, 8, 16, 16, 2>,
                     GemmPolicy<ComputeType, 8, 128, 8, 16, 32, 4>,
                     GemmPolicy<ComputeType, 8, 256, 8, 16, 16, 2>,
                     GemmPolicy<ComputeType, 8, 512, 8, 16, 16, 1>,
                     GemmPolicy<ComputeType, 16, 64, 16, 16, 16, 8>,
                     GemmPolicy<ComputeType, 16, 256, 8, 16, 16, 1>,
                     GemmPolicy<ComputeType, 16, 256, 16, 16, 16, 2>,
                     GemmPolicy<ComputeType, 16, 512, 16, 16, 16, 1>,
                     GemmPolicy<ComputeType, 32, 128, 8, 16, 32, 1>,
                     GemmPolicy<ComputeType, 32, 64, 32, 16, 16, 8>,
                     GemmPolicy<ComputeType, 32, 64, 8, 16, 16, 2>,
                     GemmPolicy<ComputeType, 32, 128, 32, 16, 16, 4>,
                     GemmPolicy<ComputeType, 32, 256, 32, 16, 16, 2>,
                     GemmPolicy<ComputeType, 32, 512, 32, 16, 16, 1>,
                     GemmPolicy<ComputeType, 64, 128, 64, 16, 16, 4>,
                     GemmPolicy<ComputeType, 64, 256, 64, 16, 16, 2>,
                     GemmPolicy<ComputeType, 64, 512, 64, 16, 16, 1>,
                     GemmPolicy<ComputeType, 128, 128, 32, 32, 32, 2>,
                     GemmPolicy<ComputeType, 128, 256, 64, 16, 16, 1>,
                     GemmPolicy<ComputeType, 128, 512, 64, 32, 16, 1>,
                     GemmPolicy<ComputeType, 256, 256, 64, 32, 16, 1>,
                     GemmPolicy<ComputeType, 256, 256, 32, 64, 16, 1>,
                     GemmPolicy<ComputeType, 256, 256, 32, 64, 32, 1>,
                     GemmPolicy<ComputeType, 128, 64, 16, 16, 64, 1>,
                     GemmPolicy<ComputeType, 128, 128, 16, 32, 64, 1>,
                     GemmPolicy<ComputeType, 128, 256, 32, 32, 16, 1>>;

}  // namespace xetla
}  // namespace gpu

#endif  // XLA_SERVICE_GPU_XETLA_GEMM_GEMM_DISPATCH_H_