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

#include "pybind11/pybind11.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/python/status_casters.h"
#include "xla/status.h"
#include "xla/util.h"

namespace py = pybind11;

namespace xla {
namespace {
Status RegisterCustomCallTarget(const PJRT_Api* c_api,
                                const std::string& fn_name, py::capsule fn,
                                int api_version) {
  static const char* const kName = "xla._CUSTOM_CALL_TARGET";
  if (std::string_view(fn.name()) != kName) {
    return InvalidArgument(
        "Argument to RegisterCustomCallTargetRegistry was not a "
        "xla._CUSTOM_CALL_TARGET capsule.");
  }

  if (c_api->extension_start == nullptr) {
    return Unimplemented(
        "The xpu plugin does not have extension in PJRT c api.");
  }
  const PJRT_Extension_Base* next =
      reinterpret_cast<const PJRT_Extension_Base*>(c_api->extension_start);
  while (next != nullptr &&
         next->type !=
             PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call) {
    next = next->next;
  }
  if (next == nullptr) {
    return Unimplemented(
        "The xpu plugin does not have a custom call extension in PJRT c api.");
  }

  PJRT_Gpu_Register_Custom_Call_Args args;
  args.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
  args.function_name = fn_name.c_str();
  args.function_name_size = fn_name.size();
  args.custom_call_function = static_cast<void*>(fn);
  args.api_version = api_version;

  RETURN_STATUS_IF_PJRT_ERROR(
      reinterpret_cast<const PJRT_Gpu_Custom_Call*>(next)->custom_call(&args),
      c_api);
  return OkStatus();
}

}  // namespace

PYBIND11_MODULE(xpu_plugin_extension, m) {
  m.def("register_custom_call_target",
        [](py::capsule c_api, const std::string& fn_name, py::capsule fn,
           const std::string& xla_platform_name, const int api_version) {
          xla::ThrowIfError(RegisterCustomCallTarget(
              static_cast<const PJRT_Api*>(c_api), fn_name, std::move(fn),
              api_version));
        },
        py::arg("c_api"), py::arg("fn_name"), py::arg("fn"),
        py::arg("xla_platform_name"), py::arg("api_version") = 0);
}
}  // namespace xla
