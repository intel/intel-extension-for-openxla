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

#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/base/const_init.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/debugging/leak_check.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/stacktrace.h"
#include "tsl/platform/status.h"
#include "tsl/platform/threadpool.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

#define RETURN_IF_SYCL_RES_ERROR(expr, ...)                            \
  do {                                                                 \
    SYCLError_t _res = (expr);                                         \
    if (ABSL_PREDICT_FALSE(_res != SYCL_SUCCESS)) {                    \
      return absl::InternalError(absl::StrCat(                         \
          __VA_ARGS__, ": ", ToString(_res))); \
    }                                                                  \
  } while (0)

namespace stream_executor {
namespace gpu {

class GpuContext {
 public:
  GpuContext(sycl::device* d, sycl::context* c) : device_(d), context_(c) {}

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

namespace {

static absl::Status InternalInit() { return absl::OkStatus(); }

}  // namespace
/* static */ absl::Status GpuDriver::Init() {
  // Cached return value from calling InternalInit()
  static absl::Status* init_retval = [] {
    return new absl::Status(InternalInit());
  }();
  return *init_retval;
}

/* static */ absl::Status GpuDriver::GetDevice(int device_ordinal,
                                              sycl::device** device) {
  auto res = SYCLGetDevice(device, device_ordinal);
  if (res == SYCL_SUCCESS) {
    return absl::OkStatus();
  }

  return absl::Status{
      absl::StatusCode::kInternal,
      absl::StrCat("failed call to syclDeviceGet: ", ToString(res))};
}

/* static */ absl::Status GpuDriver::GetDeviceName(sycl::device* device,
                                                  std::string* device_name) {
  *device_name = device->get_info<sycl::info::device::name>();
  return absl::OkStatus();
}

/* static */ absl::Status GpuDriver::CreateContext(
    int device_ordinal, sycl::device* device,
    const DeviceOptions& device_options, GpuContext** context) {
  sycl::context* sycl_context;
  SYCLGetContext(&sycl_context);
  *context = new GpuContext(device, sycl_context);
  return absl::OkStatus();
}

/* static */ void GpuDriver::DestroyContext(GpuContext* context) {
  if (context == nullptr) {
    return;
  }
  delete context;
}

/* static */ absl::Status GpuDriver::CreateGraph(GpuGraphHandle* graph) {
  return absl::UnimplementedError(
      "CreateGraph is not implemented");
}

/* static */ absl::Status GpuDriver::DestroyGraph(GpuGraphHandle graph) {
  return absl::UnimplementedError(
      "DestroyGraph is not implemented");
}

/* static */ absl::Status GpuDriver::StreamBeginCapture(
    GpuStreamHandle stream, StreamCaptureMode mode) {
  return absl::UnimplementedError(
      "StreamBeginCapture is not implemented");
}

/* static */ absl::Status GpuDriver::StreamBeginCaptureToGraph(
    GpuStreamHandle stream, GpuGraphHandle graph, StreamCaptureMode mode) {
  return absl::UnimplementedError(
      "StreamBeginCaptureToGraph is not implemented");
}

/* static */ absl::Status GpuDriver::StreamEndCapture(GpuStreamHandle stream,
                                                      GpuGraphHandle* graph) {
  return absl::UnimplementedError(
      "StreamEndCapture is not implemented");
}

/* static */ absl::Status GpuDriver::GraphInstantiate(
    GpuGraphExecHandle* exec, GpuGraphHandle graph, const GraphInstantiateFlags& flags) {
  return absl::UnimplementedError(
      "GraphInstantiate is not implemented");
}

/* static */ absl::Status GpuDriver::GraphLaunch(GpuGraphExecHandle exec,
                                                 GpuStreamHandle stream) {
  return absl::UnimplementedError(
      "GraphLaunch is not implemented");
}

/* static */ absl::Status GpuDriver::GraphNodeSetEnabled(GpuGraphExecHandle exec,
                                                         GpuGraphNodeHandle node,
                                                         bool enabled) {
  return absl::UnimplementedError(
      "GraphNodeSetEnabled is not implemented");
}

/* static */ absl::Status GpuDriver::GraphExecUpdate(
    GpuGraphExecHandle exec, GpuGraphHandle graph, GraphExecUpdateResultInfo* result) {
  return absl::UnimplementedError(
      "GraphExecUpdate is not implemented");
}

/* static */ absl::StatusOr<GpuDriver::GraphNodeType>
GpuDriver::GraphNodeGetType(GpuGraphNodeHandle node) {
  return absl::UnimplementedError(
      "GraphNodeGetType is not implemented");
}

/* static */ absl::Status GpuDriver::DestroyGraphExec(GpuGraphExecHandle exec) {
  return absl::UnimplementedError(
      "DestroyGraphExec is not implemented");
}

/* static */ absl::StatusOr<std::string> GpuDriver::GraphDebugDotPrint(
    GpuGraphHandle graph, const char* path, bool return_printed_graph) {
  return absl::UnimplementedError(
      "GraphDebugDotPrint is not implemented");
}

/* static */ absl::Status GpuDriver::DeviceGraphMemTrim(GpuDeviceHandle device) {
  return absl::UnimplementedError(
      "DeviceGraphMemTrim is not implemented");
}

/* static */ absl::StatusOr<bool> GpuDriver::StreamIsCapturing(
    GpuStreamHandle stream) {
  return absl::UnimplementedError(
      "StreamIsCapturing is not implemented");
}

/* static */ absl::Status GpuDriver::GraphConditionalHandleCreate(
    GpuGraphConditionalHandle* handle, GpuGraphHandle graph, GpuContext* context,
    unsigned int default_launch_value, unsigned int flags) {
  return absl::UnimplementedError(
      "GraphConditionalHandleCreate is not implemented");
}

/* static */ absl::StatusOr<GpuDriver::GpuGraphNodeResult>
GpuDriver::GraphAddNode(GpuGraphNodeHandle* node, GpuGraphHandle graph,
                        absl::Span<GpuGraphNodeHandle> deps,
                        const GpuGraphNodeParams& params) {
  return absl::UnimplementedError(
      "GraphAddNode is not implemented");
}

/* static */ absl::Status GpuDriver::GraphAddEmptyNode(
    GpuGraphNodeHandle* node, GpuGraphHandle graph, absl::Span<GpuGraphNodeHandle> deps) {
  return absl::UnimplementedError(
      "GraphAddEmptyNode is not implemented");
}

/* static */ absl::Status GpuDriver::GraphAddKernelNode(
    GpuGraphNodeHandle* node, GpuGraphHandle graph, absl::Span<GpuGraphNodeHandle> deps,
    absl::string_view kernel_name, GpuFunctionHandle function, unsigned int grid_dim_x,
    unsigned int grid_dim_y, unsigned int grid_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void** kernel_params, void** extra) {
  return absl::UnimplementedError(
      "GraphAddKernelNode is not implemented");
}

/*static*/ absl::Status GpuDriver::GraphExecKernelNodeSetParams(
    GpuGraphExecHandle exec, GpuGraphNodeHandle node, absl::string_view kernel_name,
    GpuFunctionHandle function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes,
    void** kernel_params, void** extra) {
  return absl::UnimplementedError(
      "GraphExecKernelNodeSetParams is not implemented");
}

/*static*/ absl::Status GpuDriver::GraphAddMemAllocNode(
    GpuGraphNodeHandle* node, GpuGraphHandle graph, absl::Span<GpuGraphNodeHandle> deps,
    GpuDriver::MemAccessFlags access_flags,
    GpuDriver::MemLocationType location_type, int device_id,
    GpuDriver::MemAllocationType allocation_type, uint64_t size,
    GpuDevicePtr* d_ptr, uint64_t max_pool_size) {
  return absl::UnimplementedError(
      "GraphAddMemAllocNode is not implemented");
}

/*static*/ absl::StatusOr<std::pair<GpuDevicePtr, uint64_t>>
GpuDriver::GraphGetMemAllocNodeParams(GpuGraphNodeHandle node) {
  return absl::UnimplementedError(
      "GraphGetMemAllocNodeParams is not implemented");
}

/*static*/ absl::Status GpuDriver::GraphAddMemFreeNode(
    GpuGraphNodeHandle* node, GpuGraphHandle graph, absl::Span<GpuGraphNodeHandle> deps,
    GpuDevicePtr gpu_dst) {
  return absl::UnimplementedError(
      "GraphAddMemFreeNode is not implemented");
}

/* static */ absl::Status GpuDriver::GraphAddMemcpyD2DNode(
    GpuContext* context, GpuGraphNodeHandle* node, GpuGraphHandle graph,
    absl::Span<GpuGraphNodeHandle> deps, GpuDevicePtr gpu_dst, GpuDevicePtr gpu_src,
    uint64_t size) {
  return absl::UnimplementedError(
      "GraphAddMemcpyD2DNode is not implemented");
}

/* static */ absl::Status GpuDriver::GraphExecMemcpyD2DNodeSetParams(
    GpuContext* context, GpuGraphExecHandle exec, GpuGraphNodeHandle node,
    GpuDevicePtr gpu_dst, GpuDevicePtr gpu_src, uint64_t size) {
  return absl::UnimplementedError(
      "GraphExecMemcpyD2DNodeSetParams is not implemented");
}

/* static */ absl::Status GpuDriver::GraphAddMemsetNode(
    GpuContext* context, GpuGraphNodeHandle* node, GpuGraphHandle graph,
    absl::Span<GpuGraphNodeHandle> deps, GpuDevicePtr dst,
    std::variant<uint8_t, uint16_t, uint32_t> bit_pattern,
    uint64_t num_elements) {
  return absl::UnimplementedError(
      "GraphAddMemsetNode is not implemented");
}

/* static */ absl::Status GpuDriver::GraphExecMemsetNodeSetParams(
    GpuContext* context, GpuGraphExecHandle exec, GpuGraphNodeHandle node, GpuDevicePtr dst,
    std::variant<uint8_t, uint16_t, uint32_t> bit_pattern,
    uint64_t num_elements) {
  return absl::UnimplementedError(
      "GraphExecMemsetNodeSetParams is not implemented");
}

/* static */ absl::Status GpuDriver::GraphAddChildNode(
    GpuGraphNodeHandle* node, GpuGraphHandle graph, absl::Span<GpuGraphNodeHandle> deps,
    GpuGraphHandle child) {
  return absl::UnimplementedError(
      "GraphAddChildNode is not implemented");
}

/*static*/ absl::Status GpuDriver::GraphExecChildNodeSetParams(GpuGraphExecHandle exec,
                                                               GpuGraphNodeHandle node,
                                                               GpuGraphHandle child) {
  return absl::UnimplementedError(
      "GraphExecChildNodeSetParams is not implemented");
}

/* static */ absl::Status GpuDriver::LaunchKernel(
    GpuContext* context, absl::string_view kernel_name, sycl::kernel* function,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes,
    sycl::queue* stream, void** kernel_params, void** extra) {
  VLOG(2) << "launching kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z;
  auto sycl_global_range =
      sycl::range<3>(block_dim_z * grid_dim_z, block_dim_y * grid_dim_y,
                     block_dim_x * grid_dim_x);
  auto sycl_local_range = sycl::range<3>(block_dim_z, block_dim_y, block_dim_x);
  sycl::nd_range<3> sycl_nd_range(
      sycl::nd_range<3>(sycl_global_range, sycl_local_range));

  stream->submit([&](auto& cgh) {
    for (uint32_t i = 0; i < static_cast<size_t*>(extra[1])[0]; i++) {
      cgh.set_arg(i, static_cast<void**>(extra[0])[i]);
    }
    cgh.parallel_for(sycl_nd_range, *function);
  });

  return absl::OkStatus();
}

/* static */ absl::Status GpuDriver::LoadPtx(GpuContext* context,
                                            const char* ptx_contents,
                                            ze_module_handle_t* module) {
  return absl::Status{absl::StatusCode::kInternal,
                     "Feature not supported on Levelzero platform (LoadPtx)"};
}

/* static */ absl::Status GpuDriver::LoadCubin(GpuContext* context,
                                              const char* cubin_bytes,
                                              ze_module_handle_t* module) {
  return absl::Status{absl::StatusCode::kInternal,
                     "Feature not supported on Levelzero platform (LoadCubin)"};
}

/* static */ absl::Status GpuDriver::LoadHsaco(GpuContext* context,
                                              const char* hsaco_contents,
                                              ze_module_handle_t* module) {
  return absl::Status{absl::StatusCode::kInternal,
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
/* static */ absl::Status GpuDriver::LoadLevelzero(
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

  return absl::OkStatus();
}

/* static */ absl::Status GpuDriver::GetModuleFunction(
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
  return absl::OkStatus();
}

/* static */ bool GpuDriver::GetModuleSymbol(GpuContext* context,
                                             ze_module_handle_t module,
                                             const char* symbol_name,
                                             void** dptr, size_t* bytes) {
  CHECK(module != nullptr && symbol_name != nullptr &&
        (*dptr != nullptr || bytes != nullptr));
  ze_result_t status =
      zeModuleGetGlobalPointer(module, symbol_name, bytes, dptr);
  if (status != ZE_RESULT_SUCCESS) {
    // symbol may not be found in the current module, but it may reside in
    // another module.
    VLOG(2) << "failed to get symbol \"" << symbol_name
            << "\" from module. L0 error: " << status;
    return false;
  }
  return true;
}

/* static */ void GpuDriver::UnloadModule(GpuContext* context,
                                          ze_module_handle_t module) {
  if (module) L0_SAFE_CALL(zeModuleDestroy(module));
}

#undef L0_SAFE_CALL

/* static */ absl::Status GpuDriver::SynchronousMemsetUint8(GpuContext* context,
                                                           void* location,
                                                           uint8_t value,
                                                           size_t size) {
  RETURN_IF_SYCL_RES_ERROR(
      SYCLMemsetD8(location, value, size, context->device()),
      "Failed to memset memory");
  return absl::OkStatus();
}

/* static */ absl::Status GpuDriver::SynchronousMemsetUint32(
    GpuContext* context, void* location, uint32_t value, size_t uint32_count) {
  RETURN_IF_SYCL_RES_ERROR(
      SYCLMemsetD32(location, value, uint32_count, context->device()),
      "Failed to memset memory");
  return absl::OkStatus();
}

/* static */ absl::Status GpuDriver::AsynchronousMemsetUint8(
    GpuContext* context, void* location, uint8_t value, size_t uint32_count,
    sycl::queue* stream) {
  RETURN_IF_SYCL_RES_ERROR(
      SYCLMemsetD8Async(location, value, uint32_count, stream),
      "Failed to enqueue async memset operation");
  return absl::OkStatus();
}

/* static */ absl::Status GpuDriver::AsynchronousMemsetUint32(
    GpuContext* context, void* location, uint32_t value, size_t uint32_count,
    sycl::queue* stream) {
  RETURN_IF_SYCL_RES_ERROR(
      SYCLMemsetD32Async(location, value, uint32_count, stream),
      "Failed to enqueue async memset operation");
  return absl::OkStatus();
}

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
    LOG(ERROR) << "could not allocate SYCL stream for context "
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
    LOG(ERROR) << "failed to destroy SYCL stream for context "
               << context->context() << ": " << ToString(res);
  } else {
    VLOG(2) << "successfully destroyed stream " << *stream << " for context "
            << context->context();
    *stream = nullptr;
  }
}

/* static */ void* GpuDriver::DeviceAllocate(GpuContext* context,
                                             uint64_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }

  void* ptr = SYCLMalloc(context->device(), bytes);
  VLOG(2) << "allocated " << ptr << " for context " << context->context()
          << " of " << bytes << " bytes";
  return ptr;
}

/* static */ void GpuDriver::DeviceDeallocate(GpuContext* context,
                                              void* location) {
  SYCLFree(context->device(), location);
  VLOG(2) << "deallocated device memory at " << location << " for context "
          << context->context();
}

/* static */ void* GpuDriver::UnifiedMemoryAllocate(GpuContext* context,
                                                    uint64_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }

  void* ptr = SYCLMallocShared(context->device(), bytes);
  VLOG(2) << "allocated " << ptr << " for context " << context->context()
          << " of " << bytes << " bytes in unified memory";
  return ptr;
}

/* static */ void GpuDriver::UnifiedMemoryDeallocate(GpuContext* context,
                                                     void* location) {
  SYCLFree(context->device(), location);
  VLOG(2) << "deallocated unified memory at " << location << " for context "
          << context->context();
}

/* static */ void* GpuDriver::HostAllocate(GpuContext* context,
                                           uint64_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }

  void* ptr = SYCLMallocHost(context->device(), bytes);
  VLOG(2) << "allocated " << ptr << " for context " << context->context()
          << " of " << bytes << " bytes";
  return ptr;
}

/* static */ void GpuDriver::HostDeallocate(GpuContext* context,
                                            void* location) {
  SYCLFree(context->device(), location);
  VLOG(2) << "deallocated host memory at " << location << " for context "
          << context->context();
}

/* static */ int GpuDriver::GetGpuStreamPriority(
    GpuContext* context, stream_executor::StreamPriority stream_priority) {
  if (stream_priority == stream_executor::StreamPriority::Default) {
    return 0;
  }
  LOG(FATAL) << "GetGpuStreamPriority not implemented on SYCL platform";
}

/* static */ absl::Status GpuDriver::InitEvent(GpuContext* context,
                                              GpuEventHandle* event_handle,
                                              EventFlags flags) {
  if (*event_handle != nullptr) {
    LOG(FATAL) << "Event is wrongly initialized before using";
  }

  *event_handle = new EventWrapper();
  (*event_handle)->event = new sycl::event;
  (*event_handle)->queue = nullptr;

  return absl::OkStatus();
}

/* static */ absl::Status GpuDriver::DestroyEvent(GpuContext* context,
                                                 GpuEventHandle* event_handle) {
  if (*event_handle == nullptr) {
    return absl::Status{absl::StatusCode::kInvalidArgument,
                       "input event cannot be null"};
  }

  delete (*event_handle)->event;
  delete (*event_handle);
  *event_handle = nullptr;

  return absl::OkStatus();
}

/* static */ absl::Status GpuDriver::RecordEvent(GpuContext* context,
                                                GpuEventHandle event_handle,
                                                GpuStreamHandle stream) {
  event_handle->queue = stream;
  *(event_handle->event) = stream->ext_oneapi_submit_barrier();

  return absl::OkStatus();
}

/* static */ bool GpuDriver::WaitStreamOnEvent(GpuContext* context,
                                               GpuStreamHandle stream,
                                               GpuEventHandle event_handle) {
  // No need to wait if it's same in-order queue.
  static absl::once_flag init_flag;
  static bool optimize_single_queue = true;

  // Env setting for debug.
  absl::call_once(init_flag, [&]() {
    const char* env = std::getenv("_XLA_OPTIMIZE_SINGLE_QUEUE");

    if (env != nullptr) {
      std::string str_value = absl::AsciiStrToLower(env);
      if (str_value == "0" || str_value == "false") {
        optimize_single_queue = false;
      }
    }
  });

  if (!optimize_single_queue || stream != event_handle->queue) {
    const std::vector<sycl::event> event_list{*(event_handle->event)};
    stream->ext_oneapi_submit_barrier(event_list);
  }

  return true;
}

/* static */ bool GpuDriver::SynchronizeContext(GpuContext* context) {
  SYCLError_t res = SYCLCtxSynchronize(context->device());
  if (res != SYCL_SUCCESS) {
    LOG(ERROR) << "could not synchronize on SYCL context: " << ToString(res)
               << " :: " << tsl::CurrentStackTrace();
    return false;
  }

  return true;
}

/* static */ absl::Status GpuDriver::SynchronizeStream(GpuContext* context,
                                                      sycl::queue* stream) {
  CHECK(stream != nullptr);
  stream->wait();
  return absl::OkStatus();
}

/* static */ bool GpuDriver::IsStreamIdle(GpuContext* context,
                                          GpuStreamHandle stream) {
  return true;
}

/* static */ absl::Status GpuDriver::SynchronousMemcpyD2H(GpuContext* context,
                                                         void* host_dst,
                                                         void* gpu_src,
                                                         uint64_t size) {
  RETURN_IF_SYCL_RES_ERROR(
      SYCLMemcpyDtoH(host_dst, gpu_src, size, context->device()),
      absl::StrFormat("failed to synchronous memcpy from device to host "
                      "host dst: %p; GPU src: %p; size: %u=0x%x",
                      host_dst, absl::bit_cast<void*>(gpu_src), size, size));
  VLOG(2) << "successfully sync memcpy'd d2h of " << size << " bytes to "
          << host_dst;
  return absl::OkStatus();
}

/* static */ absl::Status GpuDriver::SynchronousMemcpyH2D(GpuContext* context,
                                                         void* gpu_dst,
                                                         const void* host_src,
                                                         uint64_t size) {
  RETURN_IF_SYCL_RES_ERROR(
      SYCLMemcpyHtoD(gpu_dst, host_src, size, context->device()),
      absl::StrFormat(
          "failed to synchronous memcpy from host to device: GPU dst: %p;"
          " host src: %p; size: %u=0x%x",
          absl::bit_cast<void*>(gpu_dst), host_src, size, size));
  VLOG(2) << "successfully enqueued sync memcpy h2d of " << size << " bytes";
  return absl::OkStatus();
}

/* static */ absl::Status GpuDriver::SynchronousMemcpyD2D(GpuContext* context,
                                                         void* gpu_dst,
                                                         void* gpu_src,
                                                         uint64_t size) {
  RETURN_IF_SYCL_RES_ERROR(
      SYCLMemcpyDtoD(gpu_dst, gpu_src, size, context->device()),
      absl::StrFormat(
          "failed to synchronous memcpy from device to device: GPU dst: %p; "
          "GPU src: %p; size: %u=0x%x",
          absl::bit_cast<void*>(gpu_dst), absl::bit_cast<void*>(gpu_src), size,
          size));
  VLOG(2) << "successfully sync memcpy'd d2d of " << size << " bytes";
  return absl::OkStatus();
}

/* static */ bool GpuDriver::AsynchronousMemcpyD2H(GpuContext* context,
                                                   void* host_dst,
                                                   void* gpu_src, uint64_t size,
                                                   sycl::queue* stream) {
  SYCLError_t res = SYCLMemcpyDtoHAsync(host_dst, gpu_src, size, stream);
  if (res != SYCL_SUCCESS) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from device to host: %s; host dst: %p; "
        "GPU src: %p; size: %u=0x%x",
        ToString(res), host_dst, absl::bit_cast<void*>(gpu_src), size, size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << host_dst << " on stream " << stream;
  return true;
}

/* static */ bool GpuDriver::AsynchronousMemcpyH2D(GpuContext* context,
                                                   void* gpu_dst,
                                                   const void* host_src,
                                                   uint64_t size,
                                                   sycl::queue* stream) {
  SYCLError_t res = SYCLMemcpyHtoDAsync(gpu_dst, host_src, size, stream);
  if (res != SYCL_SUCCESS) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from host to device: %s; GPU dst: %p; "
        "host src: %p; size: %u=0x%x",
        ToString(res), absl::bit_cast<void*>(gpu_dst), host_src, size, size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy h2d of " << size << " bytes"
          << " from " << host_src << " to " << absl::bit_cast<void*>(gpu_dst)
          << " on stream " << stream;
  return true;
}

/* static */ bool GpuDriver::AsynchronousMemcpyD2D(GpuContext* context,
                                                   void* gpu_dst, void* gpu_src,
                                                   uint64_t size,
                                                   sycl::queue* stream) {
  SYCLError_t res = SYCLMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream);
  if (res != SYCL_SUCCESS) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from device to device: %s; GPU dst: "
        "%p; "
        "GPU src: %p; size: %u=0x%x",
        ToString(res), absl::bit_cast<void*>(gpu_dst), gpu_src, size, size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2d of " << size << " bytes"
          << " from " << gpu_src << " to " << absl::bit_cast<void*>(gpu_dst)
          << " on stream " << stream;
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

/* static */ absl::Status GpuDriver::GetComputeCapability(int* cc_major,
                                                         int* cc_minor,
                                                         sycl::device* device) {
  *cc_major = 100;
  *cc_minor = 100;
  return absl::OkStatus();
}

/* static */ absl::StatusOr<int> GpuDriver::GetMultiprocessorCount(
    sycl::device* device) {
  return device->template get_info<
             sycl::ext::intel::info::device::gpu_subslices_per_slice>() *
         device
             ->template get_info<sycl::ext::intel::info::device::gpu_slices>();
}

/* static */ absl::StatusOr<int64_t> GpuDriver::GetMaxSharedMemoryPerCore(
    sycl::device* device) {
  return device->template get_info<sycl::info::device::local_mem_size>();
}

/* static */ absl::StatusOr<int64_t> GpuDriver::GetMaxSharedMemoryPerBlock(
    sycl::device* device) {
  return device->template get_info<sycl::info::device::local_mem_size>();
}

/* static */ absl::Status GpuDriver::GetGridLimits(int* x, int* y, int* z,
                                                  sycl::device* device) {
  BlockDim block_dim_limit;
  *x = device->template get_info<
      sycl::ext::oneapi::experimental::info::device::max_work_groups<1>>();
  *y = device->template get_info<
      sycl::ext::oneapi::experimental::info::device::max_work_groups<1>>();
  *z = device->template get_info<
      sycl::ext::oneapi::experimental::info::device::max_work_groups<1>>();
  return absl::OkStatus();
}

/* static */ absl::StatusOr<int32_t> GpuDriver::GetDriverVersion() {
  int32_t version = -1;
  // TODO: Implement it for SYCL platform.
  return version;
}

/* static */ bool GpuDriver::GetDeviceMemoryInfo(GpuContext* context,
                                                 int64_t* free_out,
                                                 int64_t* total_out) {
  *free_out = -1;
  *total_out = context->device()
                   ->template get_info<sycl::info::device::global_mem_size>();
  return true;
}

/* static */ bool GpuDriver::GetDeviceTotalMemory(sycl::device* device,
                                                  uint64_t* result) {
  *result = device->get_info<sycl::info::device::global_mem_size>();
  return true;
}

/* static */ absl::StatusOr<int> GpuDriver::GetMaxOccupiedBlocksPerCore(
    GpuContext* context, GpuFunctionHandle kernel, int threads_per_block,
    size_t dynamic_shared_memory_bytes) {
  int max_blocks = 0;

  SYCLError_t result = SYCL_SUCCESS;
  // TODO(SYCL) implement this feature in HIP
  if (result != SYCL_SUCCESS) {
    return absl::Status{
        absl::StatusCode::kInternal,
        absl::StrFormat("failed to calculate occupancy of kernel %p: %s",
                        kernel, ToString(result))};
  }

  return max_blocks;
}

}  // namespace gpu
}  // namespace stream_executor
