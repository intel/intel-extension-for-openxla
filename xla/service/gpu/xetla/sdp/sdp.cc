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

#include "fmha_forward.h"
#include "xetla.hpp"

namespace gpu::xetla {

void fmha_forward_bf16(sycl::queue& q, void* query, void* key, void* value,
                       void* bias, uint8_t* dropout, float dropout_prob,
                       void* out, uint32_t num_batches, uint32_t num_heads,
                       uint32_t head_size, uint32_t num_queries,
                       uint32_t num_keys, float head_scale) {
  fmha_forward<bf16, false, false>(
      q, static_cast<bf16*>(query), static_cast<bf16*>(key),
      static_cast<bf16*>(value), static_cast<bf16*>(bias), dropout,
      dropout_prob, static_cast<bf16*>(out), num_batches, num_heads, head_size,
      num_queries, num_keys, head_scale);
}

void fmha_forward_bf16_bias(sycl::queue& q, void* query, void* key, void* value,
                            void* bias, uint8_t* dropout, float dropout_prob,
                            void* out, uint32_t num_batches, uint32_t num_heads,
                            uint32_t head_size, uint32_t num_queries,
                            uint32_t num_keys, float head_scale) {
  fmha_forward<bf16, true, false>(
      q, static_cast<bf16*>(query), static_cast<bf16*>(key),
      static_cast<bf16*>(value), static_cast<bf16*>(bias), dropout,
      dropout_prob, static_cast<bf16*>(out), num_batches, num_heads, head_size,
      num_queries, num_keys, head_scale);
}

void fmha_forward_fp16(sycl::queue& q, void* query, void* key, void* value,
                       void* bias, uint8_t* dropout, float dropout_prob,
                       void* out, uint32_t num_batches, uint32_t num_heads,
                       uint32_t head_size, uint32_t num_queries,
                       uint32_t num_keys, float head_scale) {
  fmha_forward<fp16, false, false>(
      q, static_cast<fp16*>(query), static_cast<fp16*>(key),
      static_cast<fp16*>(value), static_cast<fp16*>(bias), dropout,
      dropout_prob, static_cast<fp16*>(out), num_batches, num_heads, head_size,
      num_queries, num_keys, head_scale);
}

void fmha_forward_fp16_bias(sycl::queue& q, void* query, void* key, void* value,
                            void* bias, uint8_t* dropout, float dropout_prob,
                            void* out, uint32_t num_batches, uint32_t num_heads,
                            uint32_t head_size, uint32_t num_queries,
                            uint32_t num_keys, float head_scale) {
  fmha_forward<fp16, true, false>(
      q, static_cast<fp16*>(query), static_cast<fp16*>(key),
      static_cast<fp16*>(value), static_cast<fp16*>(bias), dropout,
      dropout_prob, static_cast<fp16*>(out), num_batches, num_heads, head_size,
      num_queries, num_keys, head_scale);
}

}  // namespace gpu::xetla