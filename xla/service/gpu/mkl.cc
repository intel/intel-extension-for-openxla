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