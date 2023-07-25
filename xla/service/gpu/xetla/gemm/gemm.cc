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

#include "hgemm_splitk.h"

namespace gpu {
namespace xetla {

#define HGEMM_IMPL_FUNC(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, B_ROW_MAJOR)                         \
  void                                                                                             \
      hgemm_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(              \
          sycl::queue& queue, sycl::half* out, const sycl::half* a,                                \
          const sycl::half* b, const int m, const int n, const int k) {                            \
    hgemm_common<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,                        \
                 B_ROW_MAJOR>(queue, out, a, b, m, n, k);                                          \
  }                                                                                                \
  void                                                                                             \
      hgemm_bias_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(         \
          sycl::queue& queue, sycl::half* out, const sycl::half* a,                                \
          const sycl::half* b, const sycl::half* bias, const int m,                                \
          const int n, const int k) {                                                              \
    hgemm_bias<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,                          \
               B_ROW_MAJOR>(queue, out, a, b, bias, m, n, k);                                      \
  }                                                                                                \
  void                                                                                             \
      hgemm_bias_res_res_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_( \
          sycl::queue& queue, sycl::half* out, const sycl::half* a,                                \
          const sycl::half* b, const sycl::half* bias, const sycl::half* res0,                     \
          const sycl::half* res1, const int m, const int n, const int k) {                         \
    hgemm_bias_res_res<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1,                     \
                       3, B_ROW_MAJOR>(queue, out, a, b, bias, res0, res1, m,                      \
                                       n, k);                                                      \
  }                                                                                                \
  void                                                                                             \
      hgemm_bias_gelu_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(    \
          sycl::queue& queue, sycl::half* out, const sycl::half* a,                                \
          const sycl::half* b, const sycl::half* bias, const int m,                                \
          const int n, const int k) {                                                              \
    hgemm_bias_gelu<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,                     \
                    B_ROW_MAJOR>(queue, out, a, b, bias, m, n, k);                                 \
  }                                                                                                \
  void                                                                                             \
      hgemm_resmul_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(       \
          sycl::queue& queue, sycl::half* out, const sycl::half* a,                                \
          const sycl::half* b, const sycl::half* mul, const int m,                                 \
          const int n, const int k) {                                                              \
    hgemm_mul<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,                           \
              B_ROW_MAJOR>(queue, out, a, b, mul, m, n, k);                                        \
  }                                                                                                \
  void                                                                                             \
      hgemm_silu_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(         \
          sycl::queue& queue, sycl::half* out, const sycl::half* a,                                \
          const sycl::half* b, const int m, const int n, const int k) {                            \
    hgemm_silu<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,                          \
               B_ROW_MAJOR>(queue, out, a, b, m, n, k);                                            \
  }                                                                                                \
  void                                                                                             \
      hgemm_res_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_##B_ROW_MAJOR##_(          \
          sycl::queue& queue, sycl::half* out, const sycl::half* a,                                \
          const sycl::half* b, const sycl::half* res, const int m,                                 \
          const int n, const int k) {                                                              \
    hgemm_res<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,                           \
              B_ROW_MAJOR>(queue, out, a, b, res, m, n, k);                                        \
  }

HGEMM_IMPL_FUNC(32, 64, 8, 16, 16, 2, true)
HGEMM_IMPL_FUNC(8, 512, 8, 16, 16, 1, true)
HGEMM_IMPL_FUNC(16, 256, 8, 16, 16, 1, true)
HGEMM_IMPL_FUNC(8, 128, 8, 16, 16, 4, true)
HGEMM_IMPL_FUNC(32, 256, 8, 32, 16, 1, true)
HGEMM_IMPL_FUNC(16, 128, 8, 16, 16, 1, true)
HGEMM_IMPL_FUNC(8, 256, 8, 32, 16, 2, true)
HGEMM_IMPL_FUNC(8, 512, 8, 32, 16, 2, true)
HGEMM_IMPL_FUNC(256, 256, 32, 64, 32, 1, true)

HGEMM_IMPL_FUNC(32, 64, 8, 16, 16, 2, false)
HGEMM_IMPL_FUNC(8, 512, 8, 16, 16, 1, false)
HGEMM_IMPL_FUNC(16, 256, 8, 16, 16, 1, false)
HGEMM_IMPL_FUNC(8, 128, 8, 16, 16, 4, false)
HGEMM_IMPL_FUNC(32, 256, 8, 32, 16, 1, false)
HGEMM_IMPL_FUNC(16, 128, 8, 16, 16, 1, false)
HGEMM_IMPL_FUNC(8, 256, 8, 32, 16, 2, false)
HGEMM_IMPL_FUNC(8, 512, 8, 32, 16, 2, false)
HGEMM_IMPL_FUNC(256, 256, 32, 64, 32, 1, false)
#if __LIBSYCL_MINOR_VERSION == 1
void hgemm_qkv_8x128_8x16x32_4(sycl::queue& queue, sycl::half* out0,
                               sycl::half* out1, sycl::half* out2,
                               const sycl::half* a, const sycl::half* b,
                               const int m, const int n, const int k) {
  hgemm_qkv<sycl::half, 8, 128, 8, 16, 32, 4, 1, 1, 3, true>(
      queue, out0, out1, out2, a, b, m, n, k);
}

void hgemm_qkv_16x256_8x16x16_1(sycl::queue& queue, sycl::half* out0,
                                sycl::half* out1, sycl::half* out2,
                                const sycl::half* a, const sycl::half* b,
                                const int m, const int n, const int k) {
  hgemm_qkv<sycl::half, 16, 256, 8, 16, 16, 1, 1, 1, 3, true>(
      queue, out0, out1, out2, a, b, m, n, k);
}

void hgemm_qkv_256x256_32x64x32_1(sycl::queue& queue, sycl::half* out0,
                                  sycl::half* out1, sycl::half* out2,
                                  const sycl::half* a, const sycl::half* b,
                                  const int m, const int n, const int k) {
  hgemm_qkv<sycl::half, 256, 256, 32, 64, 32, 1, 1, 1, 3, true>(
      queue, out0, out1, out2, a, b, m, n, k);
}
#endif
}  // namespace xetla
}  // namespace gpu
