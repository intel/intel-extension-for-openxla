/* Copyright (c) 2024 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/stream_executor/sycl/sycl_fft.h"

#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_helpers.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace stream_executor {
namespace gpu {

absl::Status SYCLFftPlan::Initialize(
    GpuExecutor *parent, Stream *stream, int rank, uint64_t *elem_count,
    uint64_t *input_embed, uint64 input_stride, uint64 input_distance,
    uint64_t *output_embed, uint64 output_stride, uint64 output_distance,
    fft::Type type, int batch_count, ScratchAllocator *scratch_allocator) {
  if (IsInitialized()) {
    return absl::InternalError("syclFFT is already initialized.");
  }

  is_initialized_ = true;
  scratch_allocator_ = scratch_allocator;
  parent_ = parent;
  fft_type_ = type;

  double scale_factor;
  std::vector<std::int64_t> dims_vec(rank, 0);
  switch (type) {
    case fft::Type::kC2CForward:
    case fft::Type::kZ2ZForward:
    case fft::Type::kR2C:
    case fft::Type::kD2Z: {
      scale_factor = 1.0 / input_distance;
      for (int i = 0; i < rank; ++i) {
        dims_vec[i] = input_embed[i];
      }
      break;
    }
    case fft::Type::kC2CInverse:
    case fft::Type::kZ2ZInverse:
    case fft::Type::kC2R:
    case fft::Type::kZ2D: {
      scale_factor = 1.0 / output_distance;
      for (int i = 0; i < rank; ++i) {
        dims_vec[i] = output_embed[i];
      }
      break;
    }
    default:
      LOG(FATAL) << "unsupported fft type";
  }

  bool is_forward = false;
  bool is_real = false;

  switch (type) {
    case fft::Type::kC2CForward:
    case fft::Type::kZ2ZForward: {
      is_forward = true;
      is_real = false;
      break;
    }
    case fft::Type::kR2C:
    case fft::Type::kD2Z: {
      is_forward = true;
      is_real = true;
      break;
    }
    case fft::Type::kC2CInverse:
    case fft::Type::kZ2ZInverse: {
      is_forward = false;
      is_real = false;
      break;
    }
    case fft::Type::kC2R:
    case fft::Type::kZ2D: {
      is_forward = false;
      is_real = true;
      break;
    }
    default:
      LOG(FATAL) << "unsupported fft type";
  }

  plan_ = std::make_unique<SYCLPlan_Helper>();
  if (is_real) {
    std::vector<int64_t> mkl_istrides(1 + rank, 0);
    std::vector<int64_t> mkl_ostrides(1 + rank, 0);
    int64_t tmp_istride = 1, tmp_ostride = 1;
    for (int64_t i = rank; i > 0; --i) {
      mkl_istrides[i] = tmp_istride;
      mkl_ostrides[i] = tmp_ostride;
      tmp_istride *= input_embed[i - 1];
      tmp_ostride *= output_embed[i - 1];
    }
    auto status = plan_->Initialize(scale_factor, dims_vec, batch_count,
                                    mkl_istrides, mkl_ostrides, tmp_istride,
                                    tmp_ostride, 0, 0, is_forward, is_real);
  } else {
    std::vector<int64_t> mkl_istrides(1, 0);
    std::vector<int64_t> mkl_ostrides(1, 0);
    auto status = plan_->Initialize(
        scale_factor, dims_vec, batch_count, mkl_istrides, mkl_ostrides, 0, 0,
        input_distance, output_distance, is_forward, is_real);
  }
  return absl::OkStatus();
}

absl::Status SYCLFftPlan::UpdateScratchAllocator(
    Stream *stream, ScratchAllocator *scratch_allocator) {
  scratch_allocator_ = scratch_allocator;

  if (scratch_size_bytes_ != 0) {
    auto allocated = scratch_allocator->AllocateBytes(scratch_size_bytes_);
    if (!allocated.ok() || (scratch_ = allocated.value()) == nullptr) {
      LOG(ERROR) << "Failed to allocate work area.";
      return allocated.status();
    }
  }
  return absl::OkStatus();
}

SYCLFftPlan::~SYCLFftPlan() {}

std::unique_ptr<fft::Plan> SYCLFft::CreateBatchedPlanWithScratchAllocator(
    Stream *stream, int rank, uint64_t *elem_count, uint64 *input_embed,
    uint64_t input_stride, uint64 input_distance, uint64 *output_embed,
    uint64_t output_stride, uint64 output_distance, fft::Type type,
    bool in_place_fft, int batch_count, ScratchAllocator *scratch_allocator) {
  auto fft_plan_ptr = std::make_unique<SYCLFftPlan>();
  absl::Status status = fft_plan_ptr->Initialize(
      parent_, stream, rank, elem_count, input_embed, input_stride,
      input_distance, output_embed, output_stride, output_distance, type,
      batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << "Initialize Params: rank: " << rank
               << " elem_count: " << *elem_count
               << " input_embed: " << *input_embed
               << " input_stride: " << input_stride
               << " input_distance: " << input_distance
               << " output_embed: " << *output_embed
               << " output_stride: " << output_stride
               << " output_distance: " << output_distance
               << " batch_count: " << batch_count;
    LOG(ERROR)
        << "Failed to initialize batched cufft plan with customized allocator: "
        << status.message();
    return nullptr;
  }
  return std::move(fft_plan_ptr);
}

void SYCLFft::UpdatePlanWithScratchAllocator(
    Stream *stream, fft::Plan *plan, ScratchAllocator *scratch_allocator) {
  SYCLFftPlan *sycl_fft_plan = dynamic_cast<SYCLFftPlan *>(plan);
  absl::Status status =
      sycl_fft_plan->UpdateScratchAllocator(stream, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to update custom allocator for syclfft plan: "
               << status.message();
  }
}

bool SYCLFft::DoFft(Stream *stream, fft::Plan *plan,
                    const DeviceMemory<std::complex<float>> &input,
                    DeviceMemory<std::complex<float>> *output) {
  SYCLFftPlan *sycl_fft_plan = dynamic_cast<SYCLFftPlan *>(plan);

  auto &plan_ = sycl_fft_plan->GetPlan();
  oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                               oneapi::mkl::dft::domain::COMPLEX>
      desc(plan_->get_dims_vec());

  desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
  desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                 plan_->get_batch_count());

  if (plan_->get_is_real()) {
    desc.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                   DFTI_COMPLEX_COMPLEX);
    desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                   (plan_->get_mkl_istrides()).data());
    desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                   (plan_->get_mkl_ostrides()).data());
    if (plan_->get_is_forward()) {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                     plan_->get_tmp_istride());
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                     plan_->get_tmp_ostride());
    } else {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                     plan_->get_tmp_ostride());
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                     plan_->get_tmp_istride());
      desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                     static_cast<float>(plan_->get_scale_factor()));
    }
  } else {
    desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                   plan_->get_input_distance());
    desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                   plan_->get_output_distance());
    desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                   static_cast<float>(plan_->get_scale_factor()));
  }
  desc.commit(*AsGpuStreamValue(stream));

  const float *in_const = reinterpret_cast<const float *>(GpuMemory(input));
  float *in = const_cast<float *>(in_const);
  float *out = reinterpret_cast<float *>(GpuMemoryMutable(output));

  ::sycl::event fft_event;
  if (plan_->get_is_forward()) {
    fft_event = oneapi::mkl::dft::compute_forward(desc, in, out);
  } else {
    fft_event = oneapi::mkl::dft::compute_backward(desc, in, out);
  }
  fft_event.wait();
  return true;
}

bool SYCLFft::DoFft(Stream *stream, fft::Plan *plan,
                    const DeviceMemory<std::complex<double>> &input,
                    DeviceMemory<std::complex<double>> *output) {
  SYCLFftPlan *sycl_fft_plan = dynamic_cast<SYCLFftPlan *>(plan);

  auto &plan_ = sycl_fft_plan->GetPlan();
  oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                               oneapi::mkl::dft::domain::COMPLEX>
      desc(plan_->get_dims_vec());

  desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
  desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                 plan_->get_batch_count());

  if (plan_->get_is_real()) {
    desc.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                   DFTI_COMPLEX_COMPLEX);
    desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                   (plan_->get_mkl_istrides()).data());
    desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                   (plan_->get_mkl_ostrides()).data());
    if (plan_->get_is_forward()) {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                     plan_->get_tmp_istride());
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                     plan_->get_tmp_ostride());
    } else {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                     plan_->get_tmp_ostride());
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                     plan_->get_tmp_istride());
      desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                     static_cast<double>(plan_->get_scale_factor()));
    }
  } else {
    desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                   plan_->get_input_distance());
    desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                   plan_->get_output_distance());
    desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                   static_cast<double>(plan_->get_scale_factor()));
  }
  desc.commit(*AsGpuStreamValue(stream));

  const double *in_const = reinterpret_cast<const double *>(GpuMemory(input));
  double *in = const_cast<double *>(in_const);
  double *out = reinterpret_cast<double *>(GpuMemoryMutable(output));

  ::sycl::event fft_event;
  if (plan_->get_is_forward()) {
    fft_event = oneapi::mkl::dft::compute_forward(desc, in, out);
  } else {
    fft_event = oneapi::mkl::dft::compute_backward(desc, in, out);
  }
  fft_event.wait();
  return true;
}

bool SYCLFft::DoFft(Stream *stream, fft::Plan *plan,
                    const DeviceMemory<float> &input,
                    DeviceMemory<std::complex<float>> *output) {
  SYCLFftPlan *sycl_fft_plan = dynamic_cast<SYCLFftPlan *>(plan);

  auto &plan_ = sycl_fft_plan->GetPlan();
  oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                               oneapi::mkl::dft::domain::REAL>
      desc(plan_->get_dims_vec());

  desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
  desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                 plan_->get_batch_count());

  if (plan_->get_is_real()) {
    desc.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                   DFTI_COMPLEX_COMPLEX);
    desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                   (plan_->get_mkl_istrides()).data());
    desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                   (plan_->get_mkl_ostrides()).data());
    if (plan_->get_is_forward()) {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                     plan_->get_tmp_istride());
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                     plan_->get_tmp_ostride());
    } else {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                     plan_->get_tmp_ostride());
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                     plan_->get_tmp_istride());
      desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                     static_cast<float>(plan_->get_scale_factor()));
    }
  } else {
    desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                   plan_->get_input_distance());
    desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                   plan_->get_output_distance());
    desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                   static_cast<float>(plan_->get_scale_factor()));
  }
  desc.commit(*AsGpuStreamValue(stream));

  const float *in_const = reinterpret_cast<const float *>(GpuMemory(input));
  float *in = const_cast<float *>(in_const);
  float *out = reinterpret_cast<float *>(GpuMemoryMutable(output));

  ::sycl::event fft_event;
  if (plan_->get_is_forward()) {
    fft_event = oneapi::mkl::dft::compute_forward(desc, in, out);
  } else {
    fft_event = oneapi::mkl::dft::compute_backward(desc, in, out);
  }
  fft_event.wait();
  return true;
}

bool SYCLFft::DoFft(Stream *stream, fft::Plan *plan,
                    const DeviceMemory<double> &input,
                    DeviceMemory<std::complex<double>> *output) {
  SYCLFftPlan *sycl_fft_plan = dynamic_cast<SYCLFftPlan *>(plan);

  auto &plan_ = sycl_fft_plan->GetPlan();
  oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                               oneapi::mkl::dft::domain::REAL>
      desc(plan_->get_dims_vec());

  desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
  desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                 plan_->get_batch_count());

  if (plan_->get_is_real()) {
    desc.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                   DFTI_COMPLEX_COMPLEX);
    desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                   (plan_->get_mkl_istrides()).data());
    desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                   (plan_->get_mkl_ostrides()).data());
    if (plan_->get_is_forward()) {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                     plan_->get_tmp_istride());
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                     plan_->get_tmp_ostride());
    } else {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                     plan_->get_tmp_ostride());
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                     plan_->get_tmp_istride());
      desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                     static_cast<double>(plan_->get_scale_factor()));
    }
  } else {
    desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                   plan_->get_input_distance());
    desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                   plan_->get_output_distance());
    desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                   static_cast<double>(plan_->get_scale_factor()));
  }
  desc.commit(*AsGpuStreamValue(stream));

  const double *in_const = reinterpret_cast<const double *>(GpuMemory(input));
  double *in = const_cast<double *>(in_const);
  double *out = reinterpret_cast<double *>(GpuMemoryMutable(output));

  ::sycl::event fft_event;
  if (plan_->get_is_forward()) {
    fft_event = oneapi::mkl::dft::compute_forward(desc, in, out);
  } else {
    fft_event = oneapi::mkl::dft::compute_backward(desc, in, out);
  }
  fft_event.wait();
  return true;
}

bool SYCLFft::DoFft(Stream *stream, fft::Plan *plan,
                    const DeviceMemory<std::complex<float>> &input,
                    DeviceMemory<float> *output) {
  SYCLFftPlan *sycl_fft_plan = dynamic_cast<SYCLFftPlan *>(plan);

  auto &plan_ = sycl_fft_plan->GetPlan();
  oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                               oneapi::mkl::dft::domain::REAL>
      desc(plan_->get_dims_vec());

  desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
  desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                 plan_->get_batch_count());

  if (plan_->get_is_real()) {
    desc.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                   DFTI_COMPLEX_COMPLEX);
    desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                   (plan_->get_mkl_istrides()).data());
    desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                   (plan_->get_mkl_ostrides()).data());
    if (plan_->get_is_forward()) {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                     plan_->get_tmp_istride());
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                     plan_->get_tmp_ostride());
    } else {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                     plan_->get_tmp_ostride());
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                     plan_->get_tmp_istride());
      desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                     static_cast<float>(plan_->get_scale_factor()));
    }
  } else {
    desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                   plan_->get_input_distance());
    desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                   plan_->get_output_distance());
    desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                   static_cast<float>(plan_->get_scale_factor()));
  }
  desc.commit(*AsGpuStreamValue(stream));

  const float *in_const = reinterpret_cast<const float *>(GpuMemory(input));
  float *in = const_cast<float *>(in_const);
  float *out = reinterpret_cast<float *>(GpuMemoryMutable(output));

  ::sycl::event fft_event;
  if (plan_->get_is_forward()) {
    fft_event = oneapi::mkl::dft::compute_forward(desc, in, out);
  } else {
    fft_event = oneapi::mkl::dft::compute_backward(desc, in, out);
  }
  fft_event.wait();
  return true;
}

bool SYCLFft::DoFft(Stream *stream, fft::Plan *plan,
                    const DeviceMemory<std::complex<double>> &input,
                    DeviceMemory<double> *output) {
  SYCLFftPlan *sycl_fft_plan = dynamic_cast<SYCLFftPlan *>(plan);

  auto &plan_ = sycl_fft_plan->GetPlan();
  oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                               oneapi::mkl::dft::domain::REAL>
      desc(plan_->get_dims_vec());

  desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
  desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                 plan_->get_batch_count());

  if (plan_->get_is_real()) {
    desc.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                   DFTI_COMPLEX_COMPLEX);
    desc.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                   (plan_->get_mkl_istrides()).data());
    desc.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                   (plan_->get_mkl_ostrides()).data());
    if (plan_->get_is_forward()) {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                     plan_->get_tmp_istride());
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                     plan_->get_tmp_ostride());
    } else {
      desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                     plan_->get_tmp_ostride());
      desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                     plan_->get_tmp_istride());
      desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                     static_cast<double>(plan_->get_scale_factor()));
    }
  } else {
    desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                   plan_->get_input_distance());
    desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                   plan_->get_output_distance());
    desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                   static_cast<double>(plan_->get_scale_factor()));
  }
  desc.commit(*AsGpuStreamValue(stream));

  const double *in_const = reinterpret_cast<const double *>(GpuMemory(input));
  double *in = const_cast<double *>(in_const);
  double *out = reinterpret_cast<double *>(GpuMemoryMutable(output));

  ::sycl::event fft_event;
  if (plan_->get_is_forward()) {
    fft_event = oneapi::mkl::dft::compute_forward(desc, in, out);
  } else {
    fft_event = oneapi::mkl::dft::compute_backward(desc, in, out);
  }
  fft_event.wait();
  return true;
}
}  // namespace gpu

void initialize_syclfft() {
  absl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::FftFactory>(
          sycl::kSyclPlatformId, "syclFFT",
          [](StreamExecutor* parent) -> fft::FftSupport * {
            gpu::GpuExecutor *sycl_executor =
                dynamic_cast<gpu::GpuExecutor *>(parent);
            if (sycl_executor == nullptr) {
              LOG(ERROR) << "Attempting to initialize an instance of the syclFFT "
                         << "support library with a non-SYCL StreamExecutor";
              return nullptr;
            }

            return new gpu::SYCLFft(sycl_executor);
          });
  if (!status.ok()) {
    LOG(ERROR) << "Unable to register syclFFT factory: " << status.message();
  }
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_syclfft, {
  stream_executor::initialize_syclfft();
});