/* Copyright (c) 2023 Intel Corporation

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

#include "xla/stream_executor/sycl/sycl_event.h"

#include "tsl/platform/statusor.h"
#include "xla/stream_executor/sycl/sycl_executor.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"
#include "xla/stream_executor/sycl/sycl_stream.h"

namespace stream_executor {
namespace gpu {

namespace sycl = ::sycl;

GpuEvent::GpuEvent(GpuExecutor* parent)
    : parent_(parent), gpu_event_(nullptr) {}

GpuEvent::~GpuEvent() {}

tsl::Status GpuEvent::Init() {
  gpu_event_ = new ITEX_GPUEvent;
  return ::tsl::OkStatus();
}

tsl::Status GpuEvent::Destroy() {
  delete gpu_event_;
  return ::tsl::OkStatus();
}

tsl::Status GpuEvent::Record(GpuStream* stream) {
  if (IsMultipleStreamEnabled()) {
    *gpu_event_ = stream->gpu_stream()->ext_oneapi_submit_barrier();
  }
  return ::tsl::OkStatus();
}

GpuEventHandle GpuEvent::gpu_event() { return gpu_event_; }

Event::Status GpuEvent::PollForStatus() {
  if (IsMultipleStreamEnabled()) {
    auto event_status =
        gpu_event_->get_info<sycl::info::event::command_execution_status>();

    switch (event_status) {
      case sycl::info::event_command_status::submitted:
      case sycl::info::event_command_status::running:
        return Event::Status::kPending;
      case sycl::info::event_command_status::complete:
        return Event::Status::kComplete;
      default:
        return Event::Status::kUnknown;
    }
  } else {
    return Event::Status::kComplete;
  }
}

}  // namespace gpu
}  // namespace stream_executor
