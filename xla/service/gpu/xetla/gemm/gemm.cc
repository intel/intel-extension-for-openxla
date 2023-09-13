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

#include "gemm.h"

#include "hgemm_impl.h"

namespace gpu {
namespace xetla {

#define HGEMM_ENUMERATE_IMPLS(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,            \
                              B_ROW_MAJOR)                                     \
  void HGEMM_ADDMM_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,             \
                             B_ROW_MAJOR)(                                     \
      sycl::queue & queue, sycl::half * out, const sycl::half* res,            \
      const sycl::half* a, const sycl::half* b, const int m, const int n,      \
      const int k, const float alpha, const float beta) {                      \
    hgemm_addmm<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,     \
                B_ROW_MAJOR>(queue, out, res, a, b, m, n, k, alpha, beta);     \
  }                                                                            \
  void HGEMM_COMMON_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,            \
                              B_ROW_MAJOR)(                                    \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const int m, const int n, const int k) {            \
    hgemm_common<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,    \
                 B_ROW_MAJOR>(queue, out, a, b, m, n, k);                      \
  }                                                                            \
  void HGEMM_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)( \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* res, const int m, const int n,    \
      const int k, const float res_factor) {                                   \
    hgemm_res<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,       \
              B_ROW_MAJOR>(queue, out, a, b, res, m, n, k, res_factor);        \
  }                                                                            \
  void HGEMM_RES_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,           \
                               B_ROW_MAJOR)(                                   \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* res0, const sycl::half* res1,     \
      const int m, const int n, const int k, const float res0_factor,          \
      const float res1_factor) {                                               \
    hgemm_res_res<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,   \
                  B_ROW_MAJOR>(queue, out, a, b, res0, res1, m, n, k,          \
                               res0_factor, res1_factor);                      \
  }                                                                            \
  void HGEMM_BIAS_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,              \
                            B_ROW_MAJOR)(                                      \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* bias, const int m, const int n,   \
      const int k, const float bias_factor) {                                  \
    hgemm_bias<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,      \
               B_ROW_MAJOR>(queue, out, a, b, bias, m, n, k, bias_factor);     \
  }                                                                            \
  void HGEMM_BIAS_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,          \
                                B_ROW_MAJOR)(                                  \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* bias, const sycl::half* res,      \
      const int m, const int n, const int k, const float bias_factor,          \
      const float res_factor) {                                                \
    hgemm_bias_res<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,  \
                   B_ROW_MAJOR>(queue, out, a, b, bias, res, m, n, k,          \
                                bias_factor, res_factor);                      \
  }                                                                            \
  void HGEMM_BIAS_RES_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,      \
                                    B_ROW_MAJOR)(                              \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* bias, const sycl::half* res0,     \
      const sycl::half* res1, const int m, const int n, const int k,           \
      const float bias_factor, const float res0_factor,                        \
      const float res1_factor) {                                               \
    hgemm_bias_res_res<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, \
                       3, B_ROW_MAJOR>(queue, out, a, b, bias, res0, res1, m,  \
                                       n, k, bias_factor, res0_factor,         \
                                       res1_factor);                           \
  }                                                                            \
  void HGEMM_BIAS_GELU_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,         \
                                 B_ROW_MAJOR)(                                 \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* bias, const int m, const int n,   \
      const int k, const float bias_factor) {                                  \
    hgemm_bias_gelu<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3, \
                    B_ROW_MAJOR>(queue, out, a, b, bias, m, n, k,              \
                                 bias_factor);                                 \
  }                                                                            \
  void HGEMM_RESMUL_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,            \
                              B_ROW_MAJOR)(                                    \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* mul, const int m, const int n,    \
      const int k) {                                                           \
    hgemm_mul<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,       \
              B_ROW_MAJOR>(queue, out, a, b, mul, m, n, k);                    \
  }                                                                            \
  void HGEMM_SILU_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,              \
                            B_ROW_MAJOR)(                                      \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const int m, const int n, const int k) {            \
    hgemm_silu<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,      \
               B_ROW_MAJOR>(queue, out, a, b, m, n, k);                        \
  }                                                                            \
  void HGEMM_QKV_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)( \
      sycl::queue & queue, sycl::half * out0, sycl::half * out1,               \
      sycl::half * out2, const sycl::half* a, const sycl::half* b,             \
      const int m, const int n, const int k) {                                 \
    hgemm_qkv<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,       \
              B_ROW_MAJOR>(queue, out0, out1, out2, a, b, m, n, k);            \
  }                                                                            \
  void HGEMM_QKV_BIAS_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,          \
                                B_ROW_MAJOR)(                                  \
      sycl::queue & queue, sycl::half * out0, sycl::half * out1,               \
      sycl::half * out2, const sycl::half* a, const sycl::half* b,             \
      const sycl::half* bias, const int m, const int n, const int k) {         \
    hgemm_qkv_bias<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,  \
                   B_ROW_MAJOR>(queue, out0, out1, out2, a, b, bias, m, n, k); \
  }

HGEMM_ENUMERATE_POLICIES(HGEMM_ENUMERATE_IMPLS)

const char* hgemm_policy_names[2 * HGEMM_NUM_POLICIES] = {
    HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_POLICY_NAME)};

int hgemm_policies_wg_mnk[2 * HGEMM_NUM_POLICIES][2]{
    HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_ENUMERATE_FUNC_TRAITS)};

void (*hgemm_addmm_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const int, const int, const int, const float,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_ADDMM_IMPL_NAME)};

void (*hgemm_common_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*, const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_COMMON_IMPL_NAME)};

void (*hgemm_res_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const int, const int, const int,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_RES_IMPL_NAME)};

void (*hgemm_res_res_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const sycl::half*, const int, const int, const int,
    const float,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_RES_RES_IMPL_NAME)};

void (*hgemm_bias_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const int, const int, const int,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_IMPL_NAME)};

void (*hgemm_bias_res_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const sycl::half*, const int, const int, const int,
    const float,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_RES_IMPL_NAME)};

void (*hgemm_bias_res_res_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const sycl::half*, const sycl::half*, const int,
    const int, const int, const float, const float, const float) = {
    HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_RES_RES_IMPL_NAME)};

void (*hgemm_bias_gelu_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const int, const int, const int,
    const float) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_BIAS_GELU_IMPL_NAME)};

void (*hgemm_resmul_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const int, const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_RESMUL_IMPL_NAME)};

void (*hgemm_silu_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*, const int,
    const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_SILU_IMPL_NAME)};

void (*hgemm_qkv_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, sycl::half*, sycl::half*, const sycl::half*,
    const sycl::half*, const int, const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_QKV_IMPL_NAME)};

void (*hgemm_qkv_bias_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, sycl::half*, sycl::half*, const sycl::half*,
    const sycl::half*, const sycl::half*, const int, const int,
    const int) = {HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_QKV_BIAS_IMPL_NAME)};

int hgemm_get_policy(hgemm_policy name, bool is_b_row_major) {
  int idx = static_cast<int>(name);
  return is_b_row_major ? idx : idx + HGEMM_NUM_POLICIES;
}

int hgemm_mapped_config(const int m, const int n, const int k,
                        const bool is_b_row_major) {
  auto it = special_mnk2policy.find(GemmShapeT{m, n, k});
  if (it != special_mnk2policy.end()) {
    int idx = it->second;
    return is_b_row_major ? idx : idx + HGEMM_NUM_POLICIES;
  }
  return -1;
}

int hgemm_qkv_mapped_config(const int m, const int n, const int k,
                            const bool is_b_row_major) {
  auto it = special_qkv_mnk2policy.find(GemmShapeT{m, n, k});
  if (it != special_qkv_mnk2policy.end()) {
    int idx = it->second;
    return is_b_row_major ? idx : idx + HGEMM_NUM_POLICIES;
  }
  return -1;
}

int select_gemm_special_config(const int m, const int n, const int k,
                               const bool is_b_row_major) {
  int policy = hgemm_mapped_config(m, n, k, is_b_row_major);
  if (policy >= 0) return policy;

  if (n == 4096 && m <= 128) {
    return hgemm_get_policy(hgemm_policy::_128x64_16x16x64_1_true_,
                            is_b_row_major);
  } else if (m >= 64) {
    if (m <= 512 && n <= 5120) {
      return hgemm_get_policy(hgemm_policy::_128x128_32x32x32_2_true_,
                              is_b_row_major);
    } else {
      return hgemm_get_policy(hgemm_policy::_256x256_64x32x16_1_true_,
                              is_b_row_major);
    }
  }

  return -1;  // let auto-config choose
}

int select_gemm_config(const int m, const int n, const int k,
                       const bool is_b_row_major, const int TOTAL_SS) {
  int idx = select_gemm_special_config(m, n, k, is_b_row_major);
  if (idx >= 0) return idx;
  std::vector<GemmMetaT> metas;
  for (int i = 0; i < HGEMM_NUM_POLICIES; i++) {
    GemmMetaT meta;
    int wg_m = hgemm_policies_wg_mnk[i][0];
    int wg_n = hgemm_policies_wg_mnk[i][1];
    int ms = (m + wg_m - 1) / wg_m;
    int ns = (n + wg_n - 1) / wg_n;
    meta.num_ss = ms * ns;
    int vm = m > wg_m ? wg_m : m;
    int vn = n > wg_n ? wg_n : n;
    meta.wg_eff = (float)vm * vn / (float)wg_m / (float)wg_n;
    meta.idx = i;
    meta.aspect_r = std::max((float)wg_m / wg_n, (float)wg_n / wg_m);
    metas.push_back(meta);
  }
  std::sort(metas.begin(), metas.end(),
            [TOTAL_SS](const auto& lhs, const auto& rhs) {
              int lss = std::abs(lhs.num_ss - TOTAL_SS);
              int rss = std::abs(rhs.num_ss - TOTAL_SS);
              if (lss != rss)
                return lss < rss;
              else if (lhs.wg_eff != rhs.wg_eff)
                return lhs.wg_eff > rhs.wg_eff;
              else
                return lhs.aspect_r < rhs.aspect_r;
            });
  idx = metas[0].idx;
  return is_b_row_major ? idx : idx + HGEMM_NUM_POLICIES;
}

}  // namespace xetla
}  // namespace gpu
