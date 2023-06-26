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

#ifndef XLA_SERVICE_GPU_MKL_H_
#define XLA_SERVICE_GPU_MKL_H_
#include <string>
#if ITEX_USE_MKL
#include "oneapi/mkl/blas.hpp"
#include "oneapi/mkl/lapack.hpp"

namespace xla {
namespace gpu {

std::string TransposeString(oneapi::mkl::transpose t);
std::string UpperLowerString(oneapi::mkl::uplo ul);
std::string DiagonalString(oneapi::mkl::diag d);
std::string SideString(oneapi::mkl::side s);

}  // namespace gpu
}  // namespace xla

#endif  // ITEX_USE_MKL
#endif  // XLA_SERVICE_GPU_MKL_H_