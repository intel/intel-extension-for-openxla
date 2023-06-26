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

#include "xla/service/gpu/mkl.h"

#include "tsl/platform/default/integral_types.h"
#include "tsl/platform/logging.h"

#if ITEX_USE_MKL
namespace xla {
namespace gpu {
std::string TransposeString(oneapi::mkl::transpose t) {
  switch (t) {
    case oneapi::mkl::transpose::N:
      return "NoTranspose";
    case oneapi::mkl::transpose::T:
      return "Transpose";
    case oneapi::mkl::transpose::C:
      return "ConjugateTranspose";
    default:
      LOG(FATAL) << "Unknown transpose " << static_cast<tsl::int32>(t);
  }
}

std::string UpperLowerString(oneapi::mkl::uplo ul) {
  switch (ul) {
    case oneapi::mkl::uplo::U:
      return "Upper";
    case oneapi::mkl::uplo::L:
      return "Lower";
    default:
      LOG(FATAL) << "Unknown upperlower " << static_cast<tsl::int32>(ul);
  }
}

std::string DiagonalString(oneapi::mkl::diag d) {
  switch (d) {
    case oneapi::mkl::diag::U:
      return "Unit";
    case oneapi::mkl::diag::N:
      return "NonUnit";
    default:
      LOG(FATAL) << "Unknown diagonal " << static_cast<tsl::int32>(d);
  }
}

std::string SideString(oneapi::mkl::side s) {
  switch (s) {
    case oneapi::mkl::side::L:
      return "Left";
    case oneapi::mkl::side::R:
      return "Right";
    default:
      LOG(FATAL) << "Unknown side " << static_cast<tsl::int32>(s);
  }
}
}  // namespace gpu
}  // namespace xla
#endif  // ITEX_USE_MKL