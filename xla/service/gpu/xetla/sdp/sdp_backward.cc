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

#include "sdp_backward.h"

#include "fmha_backward.h"
#include "xetla.hpp"

namespace gpu::xetla {

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

void fmha_backward_kernel_fp16(
    sycl::queue& q, void* query, void* key, void* value, void* out, void* bias,
    void* grad_out, void* dp_sum, void* activation_ptr, void* grad_query,
    void* grad_query_accum, void* grad_key, void* grad_value,
    uint32_t num_batches, uint32_t num_heads, uint32_t head_size,
    uint32_t num_queries, uint32_t num_keys, float head_scale) {
  const bool use_dropout = false;
  bool use_bias = bias == nullptr ? false : true;
  BOOL_SWITCH(use_bias, kUseBias, [&] {
    BOOL_SWITCH(use_dropout, kIsDropout, [&] {
      fmha_backward<fp16, kUseBias, kIsDropout>(
          q, static_cast<fp16*>(query), static_cast<fp16*>(key),
          static_cast<fp16*>(value), static_cast<fp16*>(out),
          static_cast<fp16*>(bias), static_cast<fp16*>(grad_out),
          static_cast<float*>(dp_sum), static_cast<float*>(activation_ptr),
          static_cast<fp16*>(grad_query), static_cast<float*>(grad_query_accum),
          static_cast<fp16*>(grad_key), static_cast<fp16*>(grad_value),
          num_batches, num_heads, head_size, num_queries, num_keys, head_scale);
    });
  });
}

void fmha_backward_kernel_bf16(
    sycl::queue& q, void* query, void* key, void* value, void* out, void* bias,
    void* grad_out, void* dp_sum, void* activation_ptr, void* grad_query,
    void* grad_query_accum, void* grad_key, void* grad_value,
    uint32_t num_batches, uint32_t num_heads, uint32_t head_size,
    uint32_t num_queries, uint32_t num_keys, float head_scale) {
  const bool use_dropout = false;
  bool use_bias = bias == nullptr ? false : true;
  BOOL_SWITCH(use_bias, kUseBias, [&] {
    BOOL_SWITCH(use_dropout, kIsDropout, [&] {
      fmha_backward<bf16, kUseBias, kIsDropout>(
          q, static_cast<bf16*>(query), static_cast<bf16*>(key),
          static_cast<bf16*>(value), static_cast<bf16*>(out),
          static_cast<bf16*>(bias), static_cast<bf16*>(grad_out),
          static_cast<float*>(dp_sum), static_cast<float*>(activation_ptr),
          static_cast<bf16*>(grad_query), static_cast<float*>(grad_query_accum),
          static_cast<bf16*>(grad_key), static_cast<bf16*>(grad_value),
          num_batches, num_heads, head_size, num_queries, num_keys, head_scale);
    });
  });
}

#undef BOOL_SWITCH
}  // namespace gpu::xetla