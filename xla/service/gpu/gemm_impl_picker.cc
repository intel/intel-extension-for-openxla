/* Copyright (c) 2024 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/gemm_impl_picker.h"

#include <limits>

#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/util/env_var.h"
#include "tsl/util/proto/proto_utils.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/onednn_matmul_utils.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_timer.h"
#include "xla/stream_executor/sycl/hw_info.h"

namespace xla {
namespace gpu {

namespace {

bool GetXetlaEnv() {
  bool flag = false;
  tsl::ReadBoolFromEnvVar("XETLA_GEMM", false, &flag);
  return flag;
}

bool IsXetlaSupport(const GemmConfig& config) {
  static bool flag = GetXetlaEnv();
  auto lhs_layout = MatrixLayout{config.lhs_layout};
  auto rhs_layout = MatrixLayout{config.rhs_layout};
  auto output_layout = MatrixLayout{config.output_layout};
  int64_t batch_size = output_layout.batch_size;
  bool xetla_support = flag && IsXetlaHardwareSupport() && (batch_size == 1) &&
                       (fabs(config.alpha.real() - 1.0f) < 1e-6) &&
                       output_layout.dtype != F32 &&
                       lhs_layout.dtype == output_layout.dtype;
  return xetla_support;
}

absl::StatusOr<se::gpu::BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
      return se::gpu::BlasLt::Epilogue::kDefault;
    case GemmBackendConfig::RELU:
      return se::gpu::BlasLt::Epilogue::kReLU;
    case GemmBackendConfig::GELU:
      return se::gpu::BlasLt::Epilogue::kGELU;
    case GemmBackendConfig::BIAS:
      return se::gpu::BlasLt::Epilogue::kBias;
    case GemmBackendConfig::BIAS_RELU:
      return se::gpu::BlasLt::Epilogue::kBiasThenReLU;
    case GemmBackendConfig::BIAS_GELU:
      return se::gpu::BlasLt::Epilogue::kBiasThenGELU;
    default:
      return absl::InternalError("Unsupported Epilogue.");
  }
}

absl::StatusOr<absl::Duration> GetExecuteTime(
    const HloInstruction* gemm, const AutotuneConfig& autotune_config) {
  se::DeviceMemoryAllocator* allocator = autotune_config.GetAllocator();
  TF_ASSIGN_OR_RETURN(se::Stream* const stream, autotune_config.GetStream());
  GpuBackendConfig gpu_config =
      gemm->backend_config<GpuBackendConfig>().value();
  const GemmBackendConfig& gemm_config = gpu_config.gemm_backend_config();
  const DebugOptions& debug_options =
      gemm->GetModule()->config().debug_options();

  TF_ASSIGN_OR_RETURN(GemmConfig config, GemmConfig::For(gemm));
  // Don't run autotuning concurrently on the same GPU.
  absl::MutexLock gpu_lock(&GetGpuMutex(stream->parent()));
  TF_ASSIGN_OR_RETURN(
      se::RedzoneAllocator buffer_allocator,
      AutotunerUtil::CreateRedzoneAllocator(autotune_config, debug_options));

  int64_t rng_state = 0;
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase lhs_buffer,
      AutotunerUtil::CreateBuffer(buffer_allocator, gemm->operand(0)->shape(),
                                  autotune_config, rng_state));
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase rhs_buffer,
      AutotunerUtil::CreateBuffer(buffer_allocator, gemm->operand(1)->shape(),
                                  autotune_config, rng_state));

  const Shape& output_shape =
      gemm->shape().IsTuple() ? gemm->shape().tuple_shapes(0) : gemm->shape();

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase output_buffer,
      AutotunerUtil::CreateBuffer(buffer_allocator, output_shape,
                                  autotune_config, rng_state));

  bool has_matrix_bias = config.beta != 0.;
  TF_ASSIGN_OR_RETURN(bool has_vector_bias, gpublas_lt::EpilogueAddsVectorBias(
                                                gemm_config.epilogue()));
  se::DeviceMemoryBase bias_buffer;
  if (has_vector_bias) {
    TF_ASSIGN_OR_RETURN(
        bias_buffer,
        AutotunerUtil::CreateBuffer(
            buffer_allocator, gemm->operand(has_matrix_bias ? 3 : 2)->shape(),
            autotune_config, rng_state));
  }

  TF_ASSIGN_OR_RETURN(auto epilogue, AsBlasLtEpilogue(gemm_config.epilogue()));
  se::OwningScratchAllocator<> scratch_allocator(
      stream->parent()->device_ordinal(), autotune_config.GetAllocator());

  // Run a warmup iteration without the profiler active.
  RunGemm(config, lhs_buffer, rhs_buffer, output_buffer, output_buffer,
          bias_buffer, stream, epilogue, &scratch_allocator);
  TF_ASSIGN_OR_RETURN(auto timer,
                      se::gpu::GpuTimer::Create(se::gpu::AsGpuStream(stream)));
  absl::Status status =
      RunGemm(config, lhs_buffer, rhs_buffer, output_buffer, output_buffer,
              bias_buffer, stream, epilogue, &scratch_allocator);
  if (!status.ok()) {
    return absl::InternalError("Unexpected error");
  }
  return timer.GetElapsedDuration();
}

absl::StatusOr<AutotuneResult> DoGemmAutotuneNoCache(
    const HloInstruction* gemm, const AutotuneCacheKey& key,
    const AutotuneConfig& autotune_config) {
  if (autotune_config.IsDeviceless()) {
    // Return empty result, will tune at runtime.
    return AutotuneResult{};
  }
  VLOG(3) << "Starting autotune of GemmThunk " << gemm->ToString();

  TF_ASSIGN_OR_RETURN(GemmConfig config, GemmConfig::For(gemm));
  AutotuneResult best_algorithm;

  if (!IsXetlaSupport(config)) {
    best_algorithm.mutable_gemm()->set_algorithm(se::blas::kOneDnnGemm);
    return best_algorithm;
  }

  GpuBackendConfig gpu_config =
      gemm->backend_config<GpuBackendConfig>().value();
  GemmBackendConfig updated_config = gpu_config.gemm_backend_config();
  HloInstruction* gemm_update = const_cast<HloInstruction*>(gemm);

  std::vector<AutotuneResult> results;
  std::vector<se::blas::AlgorithmType> algorithms;
  algorithms.emplace_back(se::blas::kXetlaGemm);
  algorithms.emplace_back(se::blas::kOneDnnGemm);
  for (const se::blas::AlgorithmType& algorithm : algorithms) {
    updated_config.set_selected_algorithm(algorithm);
    *gpu_config.mutable_gemm_backend_config() = updated_config;
    TF_RETURN_IF_ERROR(gemm_update->set_backend_config(gpu_config));
    absl::StatusOr<absl::Duration> run_time =
        GetExecuteTime(gemm_update, autotune_config);
    // Since there're only 2 algorithms, directly pick oneDNN if XeTLA failed.
    if (!run_time.ok()) {
      CHECK_EQ(algorithm, se::blas::kXetlaGemm);
      AutotuneResult best_algorithm;
      best_algorithm.mutable_gemm()->set_algorithm(se::blas::kOneDnnGemm);
      return best_algorithm;
    };

    results.emplace_back();
    AutotuneResult& result = results.back();
    result.mutable_gemm()->set_algorithm(algorithm);
    *result.mutable_run_time() = tsl::proto_utils::ToDurationProto(*run_time);
  }

  // Debug flag to force XeTLA path if it's available.
  bool xetla_flag = false;
  tsl::ReadBoolFromEnvVar("_FORCE_XETLA", false, &xetla_flag);
  if (xetla_flag) {
    AutotuneResult best_algorithm;
    best_algorithm.mutable_gemm()->set_algorithm(se::blas::kXetlaGemm);
    return best_algorithm;
  }

  auto best = absl::c_min_element(
      results, [](const AutotuneResult& lhs, const AutotuneResult& rhs) {
        return tsl::proto_utils::FromDurationProto(lhs.run_time()) <
               tsl::proto_utils::FromDurationProto(rhs.run_time());
      });
  return *best;
}

absl::StatusOr<bool> RunOnInstruction(HloInstruction* gemm,
                                      const AutotuneConfig& config) {
  LOG(INFO) << "Loading the autotune result of GemmThunk " << gemm->ToString();

  GpuBackendConfig gpu_config =
      gemm->backend_config<GpuBackendConfig>().value();
  GemmBackendConfig gemm_config = gpu_config.gemm_backend_config();

  // Degenerate gemms replaced with memzero operation, no need to auto tune it.
  if (gemm_config.alpha_real() == 0.0 && gemm_config.alpha_imag() == 0.0 &&
      gemm_config.beta() == 0.0) {
    VLOG(3) << "Skip degenerate gemm instruction auto tuning";
    return false;
  }

  AutotuneCacheKey key(config.GetModelStr(), *gemm);
  GemmBackendConfig updated_config = gemm_config;
  TF_ASSIGN_OR_RETURN(AutotuneResult algorithm,
                      AutotunerUtil::Autotune(gemm, config, [&] {
                        return DoGemmAutotuneNoCache(gemm, key, config);
                      }));
  updated_config.set_selected_algorithm(algorithm.gemm().algorithm());
  *gpu_config.mutable_gemm_backend_config() = updated_config;
  TF_RETURN_IF_ERROR(gemm->set_backend_config(gpu_config));
  return updated_config.SerializeAsString() != gemm_config.SerializeAsString();
}

absl::StatusOr<bool> RunOnComputation(HloComputation* computation,
                                      AutotuneConfig config) {
  bool changed = false;
  for (HloInstruction* instr : computation->instructions()) {
    if (IsCublasGemm(*instr)) {
      TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(instr, config));
      changed |= result;
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> GemmAlgorithmPicker::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrCat("GemmAlgorithmPicker for ", module->name()));
  if (module->config().debug_options().xla_gpu_autotune_level() == 0) {
    LOG(INFO)
        << "GEMM auto-tuning disabled, GemmAlgorithmPicker returning early";
    return false;
  }
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation, config_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
