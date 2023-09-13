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

namespace gpu {
namespace xetla {

// clang-format off
#define HGEMM_IMPL_NAME(HEAD, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HEAD##_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_

#define HGEMM_ADDMM_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_addmm, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_COMMON_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_common, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_RES_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_res_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_bias, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_bias_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_BIAS_RES_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_bias_res_res, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)

#define HGEMM_BIAS_GELU_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_bias_gelu, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_RESMUL_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_resmul, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_SILU_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_silu, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)

#define HGEMM_QKV_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_qkv, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
#define HGEMM_QKV_BIAS_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) HGEMM_IMPL_NAME(hgemm_qkv_bias, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)
// clang-format on

#define HGEMM_ENUMERATE_DECLS(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,            \
                              B_ROW_MAJOR)                                     \
  void HGEMM_ADDMM_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,             \
                             B_ROW_MAJOR)(                                     \
      sycl::queue & queue, sycl::half * out, const sycl::half* res,            \
      const sycl::half* a, const sycl::half* b, const int m, const int n,      \
      const int k, const float alpha, const float beta);                       \
  void HGEMM_COMMON_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,            \
                              B_ROW_MAJOR)(                                    \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const int m, const int n, const int k);             \
  void HGEMM_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)( \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* res, const int m, const int n,    \
      const int k, const float res_factor);                                    \
  void HGEMM_RES_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,           \
                               B_ROW_MAJOR)(                                   \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* res0, const sycl::half* res1,     \
      const int m, const int n, const int k, const float res0_factor,          \
      const float res1_factor);                                                \
  void HGEMM_BIAS_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,              \
                            B_ROW_MAJOR)(                                      \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* bias, const int m, const int n,   \
      const int k, const float bias_factor);                                   \
  void HGEMM_BIAS_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,          \
                                B_ROW_MAJOR)(                                  \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* bias, const sycl::half* res,      \
      const int m, const int n, const int k, const float bias_factor,          \
      const float res_factor);                                                 \
  void HGEMM_BIAS_RES_RES_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,      \
                                    B_ROW_MAJOR)(                              \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* bias, const sycl::half* res0,     \
      const sycl::half* res1, const int m, const int n, const int k,           \
      const float bias_factor, const float res0_factor,                        \
      const float res1_factor);                                                \
  void HGEMM_BIAS_GELU_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,         \
                                 B_ROW_MAJOR)(                                 \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* bias, const int m, const int n,   \
      const int k, const float bias_factor);                                   \
  void HGEMM_RESMUL_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,            \
                              B_ROW_MAJOR)(                                    \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const sycl::half* mul, const int m, const int n,    \
      const int k);                                                            \
  void HGEMM_SILU_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,              \
                            B_ROW_MAJOR)(                                      \
      sycl::queue & queue, sycl::half * out, const sycl::half* a,              \
      const sycl::half* b, const int m, const int n, const int k);             \
  void HGEMM_QKV_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)( \
      sycl::queue & queue, sycl::half * out0, sycl::half * out1,               \
      sycl::half * out2, const sycl::half* a, const sycl::half* b,             \
      const int m, const int n, const int k);                                  \
  void HGEMM_QKV_BIAS_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,          \
                                B_ROW_MAJOR)(                                  \
      sycl::queue & queue, sycl::half * out0, sycl::half * out1,               \
      sycl::half * out2, const sycl::half* a, const sycl::half* b,             \
      const sycl::half* bias, const int m, const int n, const int k);

// clang-format off
#define HGEMM_COMMA ,
#define HGEMM_NUM_POLICIES 26
#define HGEMM_ENUMERATE_POLICIES_(_, B_ROW_MAJOR, T) \
  _(8, 64, 8, 16, 32, 8, B_ROW_MAJOR)T      \
  _(8, 128, 8, 16, 16, 2, B_ROW_MAJOR)T     \
  _(8, 128, 8, 16, 32, 4, B_ROW_MAJOR)T     \
  _(8, 256, 8, 16, 16, 2, B_ROW_MAJOR)T     \
  _(8, 512, 8, 16, 16, 1, B_ROW_MAJOR)T     \
  _(16, 64, 16, 16, 16, 8, B_ROW_MAJOR)T    \
  _(16, 256, 8, 16, 16, 1, B_ROW_MAJOR)T    \
  _(16, 256, 16, 16, 16, 2, B_ROW_MAJOR)T   \
  _(16, 512, 16, 16, 16, 1, B_ROW_MAJOR)T   \
  _(32, 64, 32, 16, 16, 8, B_ROW_MAJOR)T    \
  _(32, 64, 8, 16, 16, 2, B_ROW_MAJOR)T     \
  _(32, 128, 32, 16, 16, 4, B_ROW_MAJOR)T   \
  _(32, 256, 32, 16, 16, 2, B_ROW_MAJOR)T   \
  _(32, 512, 32, 16, 16, 1, B_ROW_MAJOR)T   \
  _(64, 128, 64, 16, 16, 4, B_ROW_MAJOR)T   \
  _(64, 256, 64, 16, 16, 2, B_ROW_MAJOR)T   \
  _(64, 512, 64, 16, 16, 1, B_ROW_MAJOR)T   \
  _(128, 128, 32, 32, 32, 2, B_ROW_MAJOR)T  \
  _(128, 256, 64, 16, 16, 1, B_ROW_MAJOR)T  \
  _(128, 512, 64, 32, 16, 1, B_ROW_MAJOR)T  \
  _(256, 256, 64, 32, 16, 1, B_ROW_MAJOR)T  \
  _(256, 256, 32, 64, 16, 1, B_ROW_MAJOR)T  \
  _(256, 256, 32, 64, 32, 1, B_ROW_MAJOR)T  \
  _(128, 64, 16, 16, 64, 1, B_ROW_MAJOR)T   \
  _(128, 128, 16, 32, 64, 1, B_ROW_MAJOR)T  \
  _(128, 256, 32, 32, 16, 1, B_ROW_MAJOR)T
// clang-format on

#define HGEMM_ENUMERATE_POLICIES(_)    \
  HGEMM_ENUMERATE_POLICIES_(_, true, ) \
  HGEMM_ENUMERATE_POLICIES_(_, false, )

#define HGEMM_ENUMERATE_POLICIES_COMMA(_)         \
  HGEMM_ENUMERATE_POLICIES_(_, true, HGEMM_COMMA) \
  HGEMM_ENUMERATE_POLICIES_(_, false, HGEMM_COMMA)

HGEMM_ENUMERATE_POLICIES(HGEMM_ENUMERATE_DECLS)

#define HGEMM_POLICY_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR) \
  "_" #WG_M "x" #WG_N "_" #SG_M "x" #SG_N "x" #SG_K "_" #SLM_KS              \
  "_" #B_ROW_MAJOR "_"
#define HGEMM_POLICY_NAME_SYMBOL(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, \
                                 B_ROW_MAJOR)                          \
  _##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_

enum hgemm_policy {
  NONE = -1,
  HGEMM_ENUMERATE_POLICIES_COMMA(HGEMM_POLICY_NAME_SYMBOL)
};

#define HGEMM_ENUMERATE_FUNC_TRAITS(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, \
                                    B_ROW_MAJOR)                          \
  { WG_M, WG_N }

extern const char* hgemm_policy_names[2 * HGEMM_NUM_POLICIES];

extern int hgemm_policies_wg_mnk[2 * HGEMM_NUM_POLICIES][2];

extern void (*hgemm_addmm_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const int, const int, const int, const float,
    const float);

extern void (*hgemm_common_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*, const int,
    const int, const int);

extern void (*hgemm_res_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const int, const int, const int, const float);

extern void (*hgemm_res_res_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const sycl::half*, const int, const int, const int,
    const float, const float);

extern void (*hgemm_bias_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const int, const int, const int, const float);

extern void (*hgemm_bias_res_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const sycl::half*, const int, const int, const int,
    const float, const float);

extern void (*hgemm_bias_res_res_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const sycl::half*, const sycl::half*, const int,
    const int, const int, const float, const float, const float);

extern void (*hgemm_bias_gelu_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const int, const int, const int, const float);

extern void (*hgemm_resmul_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*,
    const sycl::half*, const int, const int, const int);

extern void (*hgemm_silu_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, const sycl::half*, const sycl::half*, const int,
    const int, const int);

extern void (*hgemm_qkv_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, sycl::half*, sycl::half*, const sycl::half*,
    const sycl::half*, const int, const int, const int);

extern void (*hgemm_qkv_bias_policies[2 * HGEMM_NUM_POLICIES])(
    sycl::queue&, sycl::half*, sycl::half*, sycl::half*, const sycl::half*,
    const sycl::half*, const sycl::half*, const int, const int, const int);

struct GemmShapeT {
  int m_, n_, k_;
  size_t operator()(const GemmShapeT& t) const {
    size_t seed = 0;
    seed ^= std::hash<int>()(t.m_) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>()(t.n_) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>()(t.k_) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
  size_t operator==(const GemmShapeT& other) const {
    return m_ == other.m_ && n_ == other.n_ && k_ == other.k_;
  }
};

static std::unordered_map<GemmShapeT, int, GemmShapeT> special_mnk2policy = {
    {{1, 4096, 16384}, hgemm_policy::_128x64_16x16x64_1_true_},
    {{1, 16384, 4096}, hgemm_policy::_8x512_8x16x16_1_true_},
    {{1, 4096, 4096}, hgemm_policy::_128x64_16x16x64_1_true_},
    {{4, 4096, 16384}, hgemm_policy::_128x64_16x16x64_1_true_},
    {{4, 16384, 4096}, hgemm_policy::_8x512_8x16x16_1_true_},
    {{4, 4096, 4096}, hgemm_policy::_128x64_16x16x64_1_true_},
    {{32, 4096, 16384}, hgemm_policy::_32x64_8x16x16_2_true_},
    {{32, 16384, 4096}, hgemm_policy::_32x512_32x16x16_1_true_},
    {{32, 4096, 4096}, hgemm_policy::_128x64_16x16x64_1_true_},
    {{33, 4096, 16384}, hgemm_policy::_128x64_16x16x64_1_true_},
    {{33, 16384, 4096}, hgemm_policy::_128x512_64x32x16_1_true_},
    {{33, 4096, 4096}, hgemm_policy::_128x64_16x16x64_1_true_},
    {{64, 4096, 16384}, hgemm_policy::_128x64_16x16x64_1_true_},
    {{64, 16384, 4096}, hgemm_policy::_128x256_64x16x16_1_true_},
    {{64, 4096, 4096}, hgemm_policy::_128x64_16x16x64_1_true_},
    {{65, 4096, 16384}, hgemm_policy::_128x64_16x16x64_1_true_},
    {{65, 16384, 4096}, hgemm_policy::_128x256_64x16x16_1_true_},
    {{65, 4096, 4096}, hgemm_policy::_128x64_16x16x64_1_true_},
    {{128, 4096, 16384}, hgemm_policy::_128x128_32x32x32_2_true_},
    {{128, 16384, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{128, 4096, 4096}, hgemm_policy::_128x128_32x32x32_2_true_},
    {{130, 4096, 16384}, hgemm_policy::_128x128_32x32x32_2_true_},
    {{130, 16384, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{130, 4096, 4096}, hgemm_policy::_128x128_16x32x64_1_true_},
    {{256, 4096, 16384}, hgemm_policy::_128x128_16x32x64_1_true_},
    {{256, 16384, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{256, 4096, 4096}, hgemm_policy::_128x128_32x32x32_2_true_},
    {{512, 4096, 16384}, hgemm_policy::_128x256_32x32x16_1_true_},
    {{512, 16384, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{512, 4096, 4096}, hgemm_policy::_128x128_32x32x32_2_true_},
    {{513, 4096, 16384}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{513, 16384, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{513, 4096, 4096}, hgemm_policy::_128x512_64x32x16_1_true_},
    {{1024, 4096, 16384}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{1024, 16384, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{1024, 4096, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{1028, 4096, 16384}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{1028, 16384, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{1028, 4096, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{2016, 4096, 16384}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{2016, 16384, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{2016, 4096, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{1, 50400, 4096}, hgemm_policy::_128x512_64x32x16_1_true_},
    {{1, 50272, 4096}, hgemm_policy::_128x512_64x32x16_1_true_},
    {{4, 50400, 4096}, hgemm_policy::_128x512_64x32x16_1_true_},
    {{4, 50272, 4096}, hgemm_policy::_128x512_64x32x16_1_true_},
    {{1, 250880, 4096}, hgemm_policy::_32x64_8x16x16_2_true_},
    {{4, 250880, 4096}, hgemm_policy::_32x64_8x16x16_2_true_},
    {{1, 11008, 4096}, hgemm_policy::_16x256_8x16x16_1_true_},
    {{4, 11008, 4096}, hgemm_policy::_16x256_8x16x16_1_true_},
    {{32, 11008, 4096}, hgemm_policy::_64x256_64x16x16_2_true_},
    {{64, 11008, 4096}, hgemm_policy::_64x256_64x16x16_2_true_},
    {{128, 11008, 4096}, hgemm_policy::_128x256_32x32x16_1_true_},
    {{256, 11008, 4096}, hgemm_policy::_256x256_32x64x32_1_true_},
    {{512, 11008, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{1024, 11008, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{2016, 11008, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{1, 32000, 4096}, hgemm_policy::_256x256_32x64x16_1_true_},
    {{4, 32000, 4096}, hgemm_policy::_256x256_32x64x16_1_true_},
    {{1, 13824, 5120}, hgemm_policy::_256x256_32x64x16_1_true_},
    {{1, 5120, 5120}, hgemm_policy::_8x128_8x16x16_2_true_},
    {{4, 13824, 5120}, hgemm_policy::_128x256_64x16x16_1_true_},
    {{4, 5120, 5120}, hgemm_policy::_8x128_8x16x16_2_true_},
    {{32, 13824, 5120}, hgemm_policy::_128x256_64x16x16_1_true_},
    {{32, 5120, 5120}, hgemm_policy::_32x128_32x16x16_4_true_},
    {{64, 13824, 5120}, hgemm_policy::_128x256_32x32x16_1_true_},
    {{64, 5120, 5120}, hgemm_policy::_128x128_16x32x64_1_true_},
    {{128, 13824, 5120}, hgemm_policy::_128x256_32x32x16_1_true_},
    {{128, 5120, 5120}, hgemm_policy::_128x128_16x32x64_1_true_},
    {{256, 13824, 5120}, hgemm_policy::_256x256_32x64x32_1_true_},
    {{256, 5120, 5120}, hgemm_policy::_128x256_32x32x16_1_true_},
    {{512, 13824, 5120}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{512, 5120, 5120}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{1024, 13824, 5120}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{1024, 5120, 5120}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{2016, 13824, 5120}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{2016, 5120, 5120}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{1, 32000, 5120}, hgemm_policy::_256x256_32x64x16_1_true_},
    {{4, 32000, 5120}, hgemm_policy::_256x256_32x64x16_1_true_},
    {{1, 7168, 14336}, hgemm_policy::_8x256_8x16x16_2_true_},
    {{1, 1792, 14336}, hgemm_policy::_16x64_16x16x16_8_true_},
    {{4, 7168, 14336}, hgemm_policy::_8x256_8x16x16_2_true_},
    {{4, 1792, 14336}, hgemm_policy::_16x64_16x16x16_8_true_},
    {{32, 7168, 14336}, hgemm_policy::_32x256_32x16x16_2_true_},
    {{32, 1792, 14336}, hgemm_policy::_16x64_16x16x16_8_true_},
    {{1, 14336, 7168}, hgemm_policy::_128x512_64x32x16_1_true_},
    {{1, 14336, 1792}, hgemm_policy::_16x256_8x16x16_1_true_},
    {{4, 14336, 7168}, hgemm_policy::_128x256_64x16x16_1_true_},
    {{4, 14336, 1792}, hgemm_policy::_128x256_32x32x16_1_true_},
    {{32, 14336, 7168}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{32, 14336, 1792}, hgemm_policy::_128x256_32x32x16_1_true_},
    {{1, 250880, 1792}, hgemm_policy::_16x256_8x16x16_1_true_},
    {{1, 2048, 8192}, hgemm_policy::_8x64_8x16x32_8_true_},
    {{1, 3584, 7168}, hgemm_policy::_32x64_8x16x16_2_true_},
    {{1, 7168, 3584}, hgemm_policy::_128x128_16x32x64_1_true_},
    {{1, 7168, 8192}, hgemm_policy::_128x128_16x32x64_1_true_},
    {{1, 8192, 2048}, hgemm_policy::_128x64_16x16x64_1_true_},
    {{1, 8192, 7168}, hgemm_policy::_8x256_8x16x16_2_true_},
    {{1, 256, 8192}, hgemm_policy::_16x64_16x16x16_8_true_},
    {{1, 32000, 2048}, hgemm_policy::_16x256_8x16x16_1_true_},
    {{4, 2048, 8192}, hgemm_policy::_8x64_8x16x32_8_true_},
    {{4, 3584, 7168}, hgemm_policy::_32x64_8x16x16_2_true_},
    {{4, 7168, 3584}, hgemm_policy::_128x128_16x32x64_1_true_},
    {{4, 7168, 8192}, hgemm_policy::_8x256_8x16x16_2_true_},
    {{4, 8192, 2048}, hgemm_policy::_8x256_8x16x16_2_true_},
    {{4, 8192, 7168}, hgemm_policy::_128x256_32x32x16_1_true_},
    {{4, 256, 8192}, hgemm_policy::_16x64_16x16x16_8_true_},
    {{4, 32000, 2048}, hgemm_policy::_256x256_32x64x16_1_true_},
    {{1024, 2048, 8192}, hgemm_policy::_128x256_32x32x16_1_true_},
    {{1024, 7168, 8192}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{1024, 8192, 2048}, hgemm_policy::_256x256_64x32x16_1_true_},
    {{1024, 8192, 7168}, hgemm_policy::_256x256_32x64x32_1_true_},
    {{1024, 256, 8192}, hgemm_policy::_32x128_32x16x16_4_true_},
};

static std::unordered_map<GemmShapeT, int, GemmShapeT> special_qkv_mnk2policy =
    {
        {{1, 4096, 16384}, hgemm_policy::_128x256_64x16x16_1_true_},
        {{1, 16384, 4096}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{1, 4096, 4096}, hgemm_policy::_128x64_16x16x64_1_true_},
        {{4, 4096, 16384}, hgemm_policy::_128x256_64x16x16_1_true_},
        {{4, 16384, 4096}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{4, 4096, 4096}, hgemm_policy::_128x64_16x16x64_1_true_},
        {{32, 4096, 16384}, hgemm_policy::_128x256_64x16x16_1_true_},
        {{32, 16384, 4096}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{32, 4096, 4096}, hgemm_policy::_128x64_16x16x64_1_true_},
        {{33, 4096, 16384}, hgemm_policy::_128x256_64x16x16_1_true_},
        {{33, 16384, 4096}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{33, 4096, 4096}, hgemm_policy::_128x64_16x16x64_1_true_},
        {{64, 4096, 16384}, hgemm_policy::_128x256_32x32x16_1_true_},
        {{64, 16384, 4096}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{64, 4096, 4096}, hgemm_policy::_128x64_16x16x64_1_true_},
        {{65, 4096, 16384}, hgemm_policy::_128x256_32x32x16_1_true_},
        {{65, 16384, 4096}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{65, 4096, 4096}, hgemm_policy::_128x64_16x16x64_1_true_},
        {{128, 4096, 16384}, hgemm_policy::_128x256_32x32x16_1_true_},
        {{128, 16384, 4096}, hgemm_policy::_128x256_64x16x16_1_true_},
        {{128, 4096, 4096}, hgemm_policy::_128x256_32x32x16_1_true_},
        {{130, 4096, 16384}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{130, 16384, 4096}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{130, 4096, 4096}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{256, 4096, 16384}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{256, 16384, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
        {{256, 4096, 4096}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{512, 4096, 16384}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{512, 16384, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
        {{512, 4096, 4096}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{513, 4096, 16384}, hgemm_policy::_128x512_64x32x16_1_true_},
        {{513, 16384, 4096}, hgemm_policy::_128x512_64x32x16_1_true_},
        {{513, 4096, 4096}, hgemm_policy::_128x512_64x32x16_1_true_},
        {{1024, 4096, 16384}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1024, 16384, 4096}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1024, 4096, 4096}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1028, 4096, 16384}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1028, 16384, 4096}, hgemm_policy::_128x512_64x32x16_1_true_},
        {{1028, 4096, 4096}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{2016, 4096, 16384}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{2016, 16384, 4096}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{2016, 4096, 4096}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1, 50400, 4096}, hgemm_policy::_16x256_8x16x16_1_true_},
        {{1, 50272, 4096}, hgemm_policy::_16x256_8x16x16_1_true_},
        {{4, 50400, 4096}, hgemm_policy::_128x512_64x32x16_1_true_},
        {{4, 50272, 4096}, hgemm_policy::_16x256_8x16x16_1_true_},
        {{1, 11008, 4096}, hgemm_policy::_32x64_8x16x16_2_true_},
        {{4, 11008, 4096}, hgemm_policy::_32x64_8x16x16_2_true_},
        {{32, 11008, 4096}, hgemm_policy::_32x64_8x16x16_2_true_},
        {{64, 11008, 4096}, hgemm_policy::_128x256_32x32x16_1_true_},
        {{128, 11008, 4096}, hgemm_policy::_128x256_32x32x16_1_true_},
        {{256, 11008, 4096}, hgemm_policy::_256x256_64x32x16_1_true_},
        {{512, 11008, 4096}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1024, 11008, 4096}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{2016, 11008, 4096}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1, 32000, 4096}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{4, 32000, 4096}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{1, 13824, 5120}, hgemm_policy::_16x256_8x16x16_1_true_},
        {{1, 5120, 5120}, hgemm_policy::_128x512_64x32x16_1_true_},
        {{4, 13824, 5120}, hgemm_policy::_128x256_32x32x16_1_true_},
        {{4, 5120, 5120}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{32, 13824, 5120}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{32, 5120, 5120}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{64, 13824, 5120}, hgemm_policy::_128x256_32x32x16_1_true_},
        {{64, 5120, 5120}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{128, 13824, 5120}, hgemm_policy::_128x256_32x32x16_1_true_},
        {{128, 5120, 5120}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{256, 13824, 5120}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{256, 5120, 5120}, hgemm_policy::_256x256_64x32x16_1_true_},
        {{512, 13824, 5120}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{512, 5120, 5120}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1024, 13824, 5120}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1024, 5120, 5120}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{2016, 13824, 5120}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{2016, 5120, 5120}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1, 32000, 5120}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{4, 32000, 5120}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{1, 7168, 14336}, hgemm_policy::_32x512_32x16x16_1_true_},
        {{1, 1792, 14336}, hgemm_policy::_8x128_8x16x16_2_true_},
        {{4, 7168, 14336}, hgemm_policy::_64x512_64x16x16_1_true_},
        {{4, 1792, 14336}, hgemm_policy::_8x128_8x16x16_2_true_},
        {{32, 7168, 14336}, hgemm_policy::_64x512_64x16x16_1_true_},
        {{32, 1792, 14336}, hgemm_policy::_64x128_64x16x16_4_true_},
        {{1, 14336, 7168}, hgemm_policy::_16x256_8x16x16_1_true_},
        {{1, 14336, 1792}, hgemm_policy::_128x256_64x16x16_1_true_},
        {{4, 14336, 7168}, hgemm_policy::_16x256_8x16x16_1_true_},
        {{4, 14336, 1792}, hgemm_policy::_16x256_8x16x16_1_true_},
        {{32, 14336, 7168}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{32, 14336, 1792}, hgemm_policy::_128x256_32x32x16_1_true_},
        {{1, 250880, 1792}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{1, 2048, 8192}, hgemm_policy::_128x128_16x32x64_1_true_},
        {{1, 3584, 7168}, hgemm_policy::_16x256_8x16x16_1_true_},
        {{1, 7168, 3584}, hgemm_policy::_64x512_64x16x16_1_true_},
        {{1, 7168, 8192}, hgemm_policy::_64x512_64x16x16_1_true_},
        {{1, 8192, 2048}, hgemm_policy::_128x256_32x32x16_1_true_},
        {{1, 8192, 7168}, hgemm_policy::_128x256_32x32x16_1_true_},
        {{1, 256, 8192}, hgemm_policy::_16x64_16x16x16_8_true_},
        {{1, 32000, 2048}, hgemm_policy::_16x256_8x16x16_1_true_},
        {{4, 2048, 8192}, hgemm_policy::_128x128_16x32x64_1_true_},
        {{4, 3584, 7168}, hgemm_policy::_16x256_8x16x16_1_true_},
        {{4, 7168, 3584}, hgemm_policy::_64x512_64x16x16_1_true_},
        {{4, 7168, 8192}, hgemm_policy::_64x512_64x16x16_1_true_},
        {{4, 8192, 2048}, hgemm_policy::_16x256_8x16x16_1_true_},
        {{4, 8192, 7168}, hgemm_policy::_128x256_32x32x16_1_true_},
        {{4, 256, 8192}, hgemm_policy::_16x64_16x16x16_8_true_},
        {{4, 32000, 2048}, hgemm_policy::_256x256_32x64x16_1_true_},
        {{1024, 2048, 8192}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1024, 7168, 8192}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1024, 8192, 2048}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1024, 8192, 7168}, hgemm_policy::_256x256_32x64x32_1_true_},
        {{1024, 256, 8192}, hgemm_policy::_128x128_32x32x32_2_true_},
};

struct GemmMetaT {
  float wg_eff;
  float num_ss;
  float aspect_r;
  int idx;
};

extern int hgemm_get_policy(hgemm_policy name, bool is_b_row_major);
extern int hgemm_mapped_config(const int m, const int n, const int k,
                               const bool is_b_row_major);
extern int hgemm_qkv_mapped_config(const int m, const int n, const int k,
                                   const bool is_b_row_major);
extern int select_gemm_special_config(const int m, const int n, const int k,
                                      const bool is_b_row_major);
extern int select_gemm_config(const int m, const int n, const int k,
                              const bool is_b_row_major,
                              const int TOTAL_SS = 64);

}  // namespace xetla
}  // namespace gpu

#endif  // XLA_SERVICE_GPU_XETLA_GEMM_H_
