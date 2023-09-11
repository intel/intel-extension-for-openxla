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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_GPU_RUNTIME_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_GPU_RUNTIME_H_

#include <string>
#include <vector>

#include "absl/strings/ascii.h"

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

enum SYCLError_t {
  SYCL_SUCCESS,
  SYCL_ERROR_NO_DEVICE,
  SYCL_ERROR_NOT_READY,
  SYCL_ERROR_INVALID_DEVICE,
  SYCL_ERROR_INVALID_POINTER,
  SYCL_ERROR_INVALID_STREAM,
  SYCL_ERROR_DESTROY_DEFAULT_STREAM,
};

typedef int DeviceOrdinal;

using SYCLDevice = sycl::device;
using SYCLStream = sycl::queue;
using SYCLEvent = sycl::event;

inline bool IsMultipleStreamEnabled() {
  bool is_multiple_stream_enabled = false;
  const char* env = std::getenv("ITEX_ENABLE_MULTIPLE_STREAM");
  if (env == nullptr) {
    return is_multiple_stream_enabled;
  }

  std::string str_value = absl::AsciiStrToLower(env);
  if (str_value == "0" || str_value == "false") {
    is_multiple_stream_enabled = false;
  } else if (str_value == "1" || str_value == "true") {
    is_multiple_stream_enabled = true;
  }

  return is_multiple_stream_enabled;
}

const char* ToString(SYCLError_t error);

SYCLError_t SYCLGetDeviceCount(int* count);

SYCLError_t SYCLGetDevice(SYCLDevice** device, int device_ordinal);

SYCLError_t SYCLGetDeviceOrdinal(const SYCLDevice& device,
                                 DeviceOrdinal* device_ordinal);

SYCLError_t SYCLGetCurrentDeviceOrdinal(DeviceOrdinal* ordinal);

SYCLError_t SYCLSetCurrentDeviceOrdinal(DeviceOrdinal ordinal);

SYCLError_t SYCLCreateStream(SYCLDevice* device_handle, SYCLStream** stream);

SYCLError_t SYCLGetDefaultStream(SYCLDevice* device_handle,
                                 SYCLStream** stream);

SYCLError_t SYCLDestroyStream(SYCLDevice* device_handle, SYCLStream* stream);

SYCLError_t SYCLGetStreamPool(SYCLDevice* device_handle,
                              std::vector<SYCLStream*>* streams);

SYCLError_t SYCLCreateEvent(SYCLDevice* device_handle, SYCLEvent* event);
SYCLError_t SYCLDestroyEvent(SYCLDevice* device_handle, SYCLEvent event);

SYCLError_t SYCLStreamWaitEvent(SYCLStream* stream, SYCLEvent event);

SYCLError_t SYCLStreamWaitStream(SYCLStream* dependent, SYCLStream* other);

SYCLError_t SYCLCtxSynchronize(SYCLDevice* device_handle);

SYCLError_t SYCLStreamSynchronize(SYCLStream* stream);

SYCLError_t SYCLMemcpyDtoH(void* dstHost, const void* srcDevice,
                           size_t ByteCount, SYCLDevice* device);

SYCLError_t SYCLMemcpyHtoD(void* dstDevice, const void* srcHost,
                           size_t ByteCount, SYCLDevice* device);

SYCLError_t SYCLMemcpyDtoD(void* dstDevice, const void* srcDevice,
                           size_t ByteCount, SYCLDevice* device);

SYCLError_t SYCLMemcpyDtoHAsync(void* dstHost, const void* srcDevice,
                                size_t ByteCount, SYCLStream* stream);

SYCLError_t SYCLMemcpyHtoDAsync(void* dstDevice, const void* srcHost,
                                size_t ByteCount, SYCLStream* stream);

SYCLError_t SYCLMemcpyDtoDAsync(void* dstDevice, const void* srcDevice,
                                size_t ByteCount, SYCLStream* stream);

SYCLError_t SYCLMemsetD8(void* dstDevice, unsigned char uc, size_t N,
                         SYCLDevice* device);

SYCLError_t SYCLMemsetD8Async(void* dstDevice, unsigned char uc, size_t N,
                              SYCLStream* stream);

SYCLError_t SYCLMemsetD32(void* dstDevice, unsigned int ui, size_t N,
                          SYCLDevice* device);

SYCLError_t SYCLMemsetD32Async(void* dstDevice, unsigned int ui, size_t N,
                               SYCLStream* stream);

void* SYCLMalloc(SYCLDevice* device, size_t ByteCount);

void* SYCLMallocHost(size_t ByteCount);

void SYCLFree(SYCLDevice* device, void* ptr);
#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_GPU_RUNTIME_H_
