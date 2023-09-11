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

#include <stdint.h>
#include <stdlib.h>

#include <map>
#include <set>
#include <utility>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/stacktrace.h"
#include "tsl/platform/static_threadlocal.h"
#include "tsl/platform/threadpool.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/platform/logging.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

namespace stream_executor {
namespace gpu {

class GpuContext {
 public:
  GpuContext(sycl::device* d) : device_(d) {
    context_ = new sycl::context(*device_);
  }
  ~GpuContext() {
    if (context_) delete context_;
  }

  sycl::device* device() const { return device_; }
  sycl::context* context() const { return context_; }

  // Disallow copying and moving.
  GpuContext(GpuContext&&) = delete;
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

 private:
  sycl::device* device_;
  sycl::context* context_;
};

/* static */ tsl::Status GpuDriver::GetDevice(int device_ordinal,
                                              SYCLDevice** device) {
  auto res = SYCLGetDevice(device, device_ordinal);
  if (res == SYCL_SUCCESS) {
    return tsl::OkStatus();
  }

  return tsl::Status{
      absl::StatusCode::kInternal,
      absl::StrCat("failed call to syclDeviceGet: ", ToString(res))};
}

/* static */ tsl::Status GpuDriver::CreateContext(
    int device_ordinal, SYCLDevice* device, const DeviceOptions& device_options,
    GpuContext** context) {
  *context = new GpuContext(device);
  return tsl::OkStatus();
}

/* static */ tsl::Status GpuDriver::LoadPtx(GpuContext* context,
                                            const char* ptx_contents,
                                            ze_module_handle_t* module) {
  return tsl::Status{absl::StatusCode::kInternal,
                     "Feature not supported on Levelzero platform (LoadPtx)"};
}

/* static */ tsl::Status GpuDriver::LoadCubin(GpuContext* context,
                                              const char* cubin_bytes,
                                              ze_module_handle_t* module) {
  return tsl::Status{absl::StatusCode::kInternal,
                     "Feature not supported on Levelzero platform (LoadCubin)"};
}

/* static */ tsl::Status GpuDriver::LoadHsaco(GpuContext* context,
                                              const char* hsaco_contents,
                                              ze_module_handle_t* module) {
  return tsl::Status{absl::StatusCode::kInternal,
                     "Feature not supported on Levelzero platform (LoadHsaco)"};
}

#define L0_SAFE_CALL(call)                 \
  {                                        \
    ze_result_t status = (call);           \
    if (status != 0) {                     \
      LOG(FATAL) << "L0 error " << status; \
      exit(1);                             \
    }                                      \
  }
/* static */ tsl::Status GpuDriver::LoadLevelzero(
    GpuContext* context, const char* spir_contents, const size_t size,
    ze_module_handle_t* ze_module) {
  const sycl::context* sycl_context = context->context();
  const sycl::device* sycl_device = context->device();
  auto ze_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*sycl_device);
  auto ze_context =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*sycl_context);

  ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                 nullptr,
                                 ZE_MODULE_FORMAT_IL_SPIRV,
                                 size,
                                 (const uint8_t*)spir_contents,
                                 nullptr,
                                 nullptr};

  ze_module_build_log_handle_t buildlog;
  ze_result_t status =
      zeModuleCreate(ze_context, ze_device, &moduleDesc, ze_module, &buildlog);
  if (status != 0) {
    size_t szLog = 0;
    zeModuleBuildLogGetString(buildlog, &szLog, nullptr);

    std::unique_ptr<char> PLogs(new char[szLog]);
    zeModuleBuildLogGetString(buildlog, &szLog, PLogs.get());
    std::string PLog(PLogs.get());
    LOG(FATAL) << "L0 error " << status << ": " << PLog;
  }

  return ::tsl::OkStatus();
}

/* static */ tsl::Status GpuDriver::GetModuleFunction(
    GpuContext* context, ze_module_handle_t module, const char* kernel_name,
    sycl::kernel** sycl_kernel) {
  const sycl::context* sycl_context = context->context();
  CHECK(module != nullptr && kernel_name != nullptr);
  ze_kernel_handle_t ze_kernel;
  std::string kernel_name_fix = std::string(kernel_name);
  ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                                 kernel_name_fix.c_str()};

  if (VLOG_IS_ON(2)) {
    bool First = true;
    std::string PINames{""};
    uint32_t Count = 0;
    L0_SAFE_CALL(zeModuleGetKernelNames(module, &Count, nullptr));
    std::unique_ptr<const char*[]> PNames(new const char*[Count]);
    L0_SAFE_CALL(zeModuleGetKernelNames(module, &Count, PNames.get()));
    for (uint32_t I = 0; I < Count; ++I) {
      PINames += (!First ? ";" : "");
      PINames += PNames[I];
      First = false;
    }
    VLOG(2) << "Required kernel name: " << kernel_name;
    VLOG(2) << "L0 Module has kernel: " << PINames;
  }
  L0_SAFE_CALL(zeKernelCreate(module, &kernelDesc, &ze_kernel));

  sycl::kernel_bundle<sycl::bundle_state::executable> kernel_bundle =
      sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                               sycl::bundle_state::executable>({module},
                                                               *sycl_context);
  auto kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {kernel_bundle, ze_kernel}, *sycl_context);
  *sycl_kernel = new sycl::kernel(kernel);
  return tsl::OkStatus();
}
#undef L0_SAFE_CALL

/* static */ bool GpuDriver::CreateStream(GpuContext* context,
                                          sycl::queue** stream, int priority) {
  SYCLError_t res;
  if (priority == 0) {
    res = SYCLCreateStream(context->device(), stream);
  } else {
    LOG(ERROR)
        << "CreateStream with priority is not supported on SYCL platform";
  }

  if (res != SYCL_SUCCESS) {
    LOG(ERROR) << "could not allocate CUDA stream for context "
               << context->context() << ": " << ToString(res);
    return false;
  }

  VLOG(2) << "successfully created stream " << *stream << " for context "
          << context->context() << " on thread";
  return true;
}

/* static */ void GpuDriver::DestroyStream(GpuContext* context,
                                           sycl::queue** stream) {
  if (*stream == nullptr) {
    return;
  }
  SYCLError_t res = SYCLDestroyStream(context->device(), *stream);

  if (res != SYCL_SUCCESS) {
    LOG(ERROR) << "failed to destroy CUDA stream for context "
               << context->context() << ": " << ToString(res);
  } else {
    VLOG(2) << "successfully destroyed stream " << *stream << " for context "
            << context->context();
    *stream = nullptr;
  }
}
#if 0
/* static */ void* GpuDriver::DeviceAllocate(GpuContext* context,
                                             uint64_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }

  ScopedActivateContext activated{context};
  CUdeviceptr result = 0;
  CUresult res = cuMemAlloc(&result, bytes);
  if (res != CUDA_SUCCESS) {
    // LOG(INFO) because this isn't always important to users (e.g. BFCAllocator
    // implements a retry if the first allocation fails).
    LOG(INFO) << "failed to allocate "
              << tsl::strings::HumanReadableNumBytes(bytes) << " (" << bytes
              << " bytes) from device: " << ToString(res);
    return nullptr;
  }
  void* ptr = reinterpret_cast<void*>(result);
  VLOG(2) << "allocated " << ptr << " for context " << context->context()
          << " of " << bytes << " bytes";
  return ptr;
}

/* static */ void GpuDriver::DeviceDeallocate(GpuContext* context,
                                              void* location) {
  ScopedActivateContext activation(context);
  CUdeviceptr pointer = absl::bit_cast<CUdeviceptr>(location);
  CUresult res = cuMemFree(pointer);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "failed to free device memory at " << location
               << "; result: " << ToString(res);
  } else {
    VLOG(2) << "deallocated " << location << " for context "
            << context->context();
  }
}
#endif
/* static */ void* GpuDriver::UnifiedMemoryAllocate(GpuContext* context,
                                                    uint64_t bytes) {
  auto ptr = aligned_alloc_shared(64, bytes, *(context->device()),
                                  *(context->context()));
  return ptr;
}

/* static */ void GpuDriver::UnifiedMemoryDeallocate(GpuContext* context,
                                                     void* location) {
  sycl::free(location, *(context->context()));
}

/* static */ void* GpuDriver::HostAllocate(GpuContext* context,
                                           uint64_t bytes) {
  void* host_mem = aligned_alloc_host(64, bytes, *(context->context()));
  return host_mem;
}

/* static */ void GpuDriver::HostDeallocate(GpuContext* context,
                                            void* location) {
  sycl::free(location, *(context->context()));
}

/* static */ int GpuDriver::GetGpuStreamPriority(
    GpuContext* context, stream_executor::StreamPriority stream_priority) {
  // ScopedActivateContext activation(context);
  if (stream_priority == stream_executor::StreamPriority::Default) {
    return 0;
  }
  LOG(FATAL) << "GetGpuStreamPriority not implemented on SYCL platform";
}

/* static */ tsl::Status GpuDriver::InitEvent(GpuContext* context,
                                              GpuEventHandle* event,
                                              EventFlags flags) {
  *event = new SYCLEvent;
  return tsl::OkStatus();
}

/* static */ tsl::Status GpuDriver::DestroyEvent(GpuContext* context,
                                                 GpuEventHandle* event) {
  if (*event == nullptr) {
    return tsl::Status{absl::StatusCode::kInvalidArgument,
                       "input event cannot be null"};
  }

  delete (*event);
  return tsl::OkStatus();
}

/* static */ tsl::Status GpuDriver::RecordEvent(GpuContext* context,
                                                GpuEventHandle event,
                                                GpuStreamHandle stream) {
  if (IsMultipleStreamEnabled()) {
    *event = stream->ext_oneapi_submit_barrier();
  }
  return tsl::OkStatus();
}

/* static */ bool GpuDriver::IsStreamIdle(GpuContext* context,
                                          GpuStreamHandle stream) {
  return true;
}

/* static */ int GpuDriver::GetDeviceCount() {
  int device_count = 0;
  SYCLError_t res = SYCLGetDeviceCount(&device_count);
  if (res != SYCL_SUCCESS) {
    LOG(ERROR) << "could not retrieve SYCL device count: " << ToString(res);
    return 0;
  }

  return device_count;
}

}  // namespace gpu
}  // namespace stream_executor