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

#include "xla/stream_executor/sycl/hw_info.h"

#include <string>

#define XE_MASK 0xff0
#define ARC_MASK 0xff00

const int32_t XeHPC_id = 0xbd0;
const int32_t XeHPC_id_2 = 0xb60;

// PVC 1550VG does not have XeMatrix engine, we distinguish it from other PVCs
// by device id.
const int32_t XeHPC_no_xmx_id = 0xbd4;

const int32_t ARC_id = 0x5600;

bool IsXeHPC(const sycl::device* device_ptr) {
  if (device_ptr == nullptr) {
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
      auto device_list = platform.get_devices();
      for (const auto& device : device_list) {
        if (device.is_gpu()) {
          auto id =
              device.get_info<sycl::ext::intel::info::device::device_id>();
          if ((id & XE_MASK) == XeHPC_id || (id & XE_MASK) == XeHPC_id_2) {
            return true;
          }
        }
      }
    }
  } else {
    auto id = device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
    if ((id & XE_MASK) == XeHPC_id || (id & XE_MASK) == XeHPC_id_2) {
      return true;
    }
  }
  return false;
}

// TODO(intel): use sycl api like `devices.has(sycl::aspect::ext_intel_matrix)`
// instead of device id once compiler supports XMX query interface.
bool HasXMX(const sycl::device* device_ptr) {
  if (device_ptr == nullptr) {
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
      auto device_list = platform.get_devices();
      for (const auto& device : device_list) {
        if (device.is_gpu()) {
          auto id =
              device.get_info<sycl::ext::intel::info::device::device_id>();
          if (IsXeHPC(&device)) {
            if (id == XeHPC_no_xmx_id) {
              return false;
            } else {
              return true;
            }
          }
        }
      }
    }
  } else {
    auto id = device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
    if (IsXeHPC(device_ptr)) {
      if (id == XeHPC_no_xmx_id) {
        return false;
      } else {
        return true;
      }
    }
  }
  return false;
}

bool IsXetlaHardwareSupport() {
  static bool flag = IsXeHPC(nullptr) && HasXMX(nullptr);
  return flag;
}

bool IsXeHPG(const sycl::device* device_ptr) { return IsARC(device_ptr); }

bool IsARC(const sycl::device* device_ptr) {
  if (device_ptr == nullptr) {
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
      auto device_list = platform.get_devices();
      for (const auto& device : device_list) {
        if (device.is_gpu()) {
          auto id =
              device.get_info<sycl::ext::intel::info::device::device_id>();
          if ((id & ARC_MASK) == ARC_id) {
            return true;
          }
        }
      }
    }
  } else {
    auto id = device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
    if ((id & ARC_MASK) == ARC_id) {
      return true;
    }
  }
  return false;
}

uint64_t GetMaxAllocateLimitByte(sycl::device* device_ptr) {
  uint64_t limit = std::numeric_limits<uint64_t>::max();
  if (device_ptr == nullptr) {
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
      auto device_list = platform.get_devices();
      for (const auto& device : device_list) {
        if (device.is_gpu()) {
          limit = std::min(
              limit, device.get_info<sycl::info::device::max_mem_alloc_size>());
        }
      }
    }
  } else {
    limit = device_ptr->get_info<sycl::info::device::max_mem_alloc_size>();
  }
  return limit;
}
