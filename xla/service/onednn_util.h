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

#ifndef XLA_SERVICE_ONEDNN_UTIL_H_
#define XLA_SERVICE_ONEDNN_UTIL_H_

#include <map>
#include <string>
#include <utility>
#include <vector>
#include "dnnl.hpp"       // NOLINT(build/include_subdir)
#include "dnnl_sycl.hpp"  // NOLINT(build/include_subdir)
#include "tsl/util/env_var.h"
#include "xla/stream_executor/gpu/gpu_types.h"

namespace xla {
inline dnnl::memory::dims CalculateTFStrides(
    const dnnl::memory::dims& dims_tf_order) {
  CHECK_GT(dims_tf_order.size(), 0);
  dnnl::memory::dims strides(dims_tf_order.size(), 1);
  for (int d = strides.size() - 2; d >= 0; d--) {
    strides[d] = strides[d + 1] * dims_tf_order[d + 1];
  }
  return strides;
}

static dnnl::engine& FindOrCreateEngine(se::gpu::GpuStreamHandle stream) {
  static std::map<se::gpu::GpuStreamHandle, dnnl::engine> stream_engine_map;
  auto iter = stream_engine_map.find(stream);
  if (iter != stream_engine_map.end()) return iter->second;

  dnnl::engine engine;
  engine = dnnl::sycl_interop::make_engine(stream->get_device(),
                                           stream->get_context());
  return stream_engine_map
      .insert(std::pair<se::gpu::GpuStreamHandle, dnnl::engine>(stream, engine))
      .first->second;
}

inline dnnl::fpmath_mode GetFP32MathMode() {
  std::string fp32_math_mode = "fp32";
  TF_CHECK_OK(tsl::ReadStringFromEnvVar("ITEX_FP32_MATH_MODE", "fp32",
                                        &fp32_math_mode));
  fp32_math_mode = tsl::str_util::Lowercase(fp32_math_mode);
  if (fp32_math_mode == "fp32") {
    return dnnl::fpmath_mode::strict;
  }
  if (fp32_math_mode == "tf32") {
    return dnnl::fpmath_mode::tf32;
  }
  if (fp32_math_mode == "bf32") {
    LOG(FATAL) << "Did not support BF32 math mode on GPU ";
  }
  LOG(FATAL)
      << "Invalid ITEX_FP32_MATH_MODE, should be FP32, TF32 or BF32, but got "
      << fp32_math_mode;
}

inline dnnl::memory CreateDnnlMemory(const dnnl::memory::desc& md,
                                     const dnnl::engine& engine,
                                     void* data_handle = nullptr) {
  if (engine.get_kind() == dnnl::engine::kind::gpu) {
    auto kind = dnnl::sycl_interop::memory_kind::usm;
    if (data_handle == nullptr)
      return dnnl::sycl_interop::make_memory(md, engine, kind,
                                             DNNL_MEMORY_ALLOCATE);
    else
      return dnnl::sycl_interop::make_memory(md, engine, kind, data_handle);
  }

  // Default path, always assume it's CPU engine.
  CHECK(engine.get_kind() == dnnl::engine::kind::cpu)
      << "Create oneDNN memory for unsupported engine.";
  if (data_handle == nullptr)
    return dnnl::memory(md, engine);
  else
    return dnnl::memory(md, engine, data_handle);
}
}  // namespace xla
#endif  // XLA_SERVICE_ONEDNN_UTIL_H_