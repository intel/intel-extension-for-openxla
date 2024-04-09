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

#ifndef XLA_SERVICE_GPU_UTILS
#define XLA_SERVICE_GPU_UTILS

#include <sycl/sycl.hpp>

#define UNROLL_ON_DEVICE _Pragma("unroll")

// Represents an aligned array of N elements of T. Data pointers can be
// reinterpreted as this type to generate vectorized loads/stores in a kernel.
template <typename T, uint32_t N, typename Func = sycl::plus<T>>
class alignas(alignof(T) * N) AlignedVector {
 public:
  AlignedVector() = default;

  explicit AlignedVector(T uniform) {
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < N; ++i) { values_[i] = uniform; }
  }

  template <typename U>
  void Load(const AlignedVector<U, N>& other) {
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < N; ++i) {
      values_[i] = static_cast<T>(other[i]);
    }
  }

  template <typename U>
  void Accumulate(const AlignedVector<U, N>& other) {
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < N; ++i) {
      values_[i] = Func()(values_[i], static_cast<T>(other[i]));
    }
  }

  template <typename U>
  void Store(AlignedVector<U, N>& other) {
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < N; ++i) {
      other[i] = static_cast<U>(values_[i]);
    }
  }

  template <typename U>
  void PartialStore(AlignedVector<U, N>& other, uint32_t num,
                    uint32_t offset = 0) {
    UNROLL_ON_DEVICE for (uint32_t i = 0; i < N && i < num; ++i) {
      other[i] = static_cast<U>(values_[i + offset]);
    }
  }

  T& operator[](uint32_t i) { return values_[i]; }
  const T& operator[](uint32_t i) const { return values_[i]; }

 private:
  T values_[N];
};

#undef UNROLL_ON_DEVICE

#endif  //  XLA_SERVICE_GPU_UTILS
