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