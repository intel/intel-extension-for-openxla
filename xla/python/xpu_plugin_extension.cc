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

#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/python/py_client_gpu.h"
#include "xla/pjrt/status_casters.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/util.h"

namespace nb = nanobind;

namespace xla {
namespace {
absl::Status RegisterCustomCallTarget(const PJRT_Api* c_api,
                                const char* fn_name_c_str, size_t fn_name_size,
                                nb::capsule fn, int api_version,
                                XLA_FFI_Handler_Traits traits) {
  if (c_api->extension_start == nullptr) {
    return Unimplemented("The plugin does not have extension.");
  }
  const PJRT_Extension_Base* next =
      reinterpret_cast<const PJRT_Extension_Base*>(c_api->extension_start);
  while (next != nullptr &&
         next->type !=
             PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call) {
    next = next->next;
  }
  if (next == nullptr) {
    return Unimplemented("The plugin does not have a custom call extension.");
  }

  if (traits != 0) {
    return Unimplemented("The plugin does not support custom call traits.");
  }

  PJRT_Gpu_Register_Custom_Call_Args args;
  args.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
  args.function_name = fn_name_c_str;
  args.function_name_size = fn_name_size;
  args.api_version = api_version;
  args.custom_call_function = static_cast<void*>(fn.data());
  RETURN_STATUS_IF_PJRT_ERROR(
      reinterpret_cast<const PJRT_Gpu_Custom_Call*>(next)->custom_call(&args),
      c_api);
  return absl::OkStatus();
}

template <typename T>
nb::capsule EncapsulateFunction(T* fn) {
  return nb::capsule(absl::bit_cast<void*>(fn),
                           "xla._CUSTOM_CALL_TARGET");
}

nb::dict Registrations() {
  nb::dict dict;
  dict["xla_python_gpu_callback"] =
      EncapsulateFunction(xla::XlaPythonGpuCallback);
  return dict;
}

}  // namespace

NB_MODULE(xpu_plugin_extension, m) {
  tsl::ImportNumpy();
  m.def(
      "register_custom_call_target",
      [](nb::capsule c_api, nb::object fn_name_py, nb::capsule fn,
         nb::str xla_platform_name, int api_version,
         XLA_FFI_Handler_Traits traits) {
        const char* fn_name_c_str;
        size_t fn_name_size;
        nb::str fn_name_bn_str;
        if (nb::try_cast<nb::str>(fn_name_py, fn_name_bn_str)) {
          fn_name_c_str = fn_name_bn_str.c_str();
          fn_name_size = nb::len(fn_name_bn_str);
        } else{
          nb::bytes bytes = nb::cast<nb::bytes>(fn_name_py);
          fn_name_c_str = bytes.c_str();
          fn_name_size = bytes.size();
        }
        xla::ThrowIfError(RegisterCustomCallTarget(
            static_cast<const PJRT_Api*>(c_api.data()), fn_name_c_str,
            fn_name_size, std::move(fn), api_version, traits));
      },
      nb::arg("c_api"), nb::arg("fn_name"), nb::arg("fn"),
      nb::arg("xla_platform_name"), nb::arg("api_version") = 0,
      nb::arg("traits") = 0);
  m.def("registrations", &Registrations);
}
}  // namespace xla
