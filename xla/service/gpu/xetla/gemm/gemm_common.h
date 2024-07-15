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

#ifndef XLA_SERVICE_GPU_XETLA_GEMM_GEMM_COMMON_H_
#define XLA_SERVICE_GPU_XETLA_GEMM_GEMM_COMMON_H_

#include "xla/service/gpu/matrix_descriptor.h"

namespace se = ::stream_executor;
const int kMaxNumEpilogues = 4;

namespace gpu {
namespace xetla {

enum EpilogueType {
  BIAS = 0,
  RES_ADD,
  GELU,
  RES_MUL,
  SILU,
};

}  // namespace xetla
}  // namespace gpu

#endif  // XLA_SERVICE_GPU_XETLA_GEMM_GEMM_COMMON_H_