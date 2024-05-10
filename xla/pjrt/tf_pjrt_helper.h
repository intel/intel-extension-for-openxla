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

#ifndef XLA_PJRT_TF_PJRT_HELPER_H_
#define XLA_PJRT_TF_PJRT_HELPER_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xla/pjrt/tf_xpu_pjrt_client.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/custom_call_target_registry.h"


// expose C api for third party libraries like Horovod which needs to get stream
// and pjrt_buffer.
extern "C" {
typedef struct PjRtBuffer_Info PjRtBuffer_Info;
void* C_ITEXOpaqueDataPointerFromPjRtBuffer(PJRT_Buffer* pjrt_c_buffer);
PJRT_Buffer* C_ITEXCreatePjRtBuffer(int device_id,
                                    PjRtBuffer_Info* pjrt_buffer_info,
                                    PJRT_Client* pjrt_c_client);
PJRT_Buffer* C_ITEXCreateSEPjRtBuffer(int device_id,
                                      PjRtBuffer_Info* pjrt_buffer_info,
                                      PJRT_Client* pjrt_c_client);
void* C_ITEXGetStreamFromPjRtDevice(int device_id, PJRT_Client* pjrt_c_client);
void C_RegisterCustomCallTarget(const char* symbol, void* address,
                                const char* platform);
}

xla::PrimitiveType XlaDataTypeFromString(std::string data_type);

void* ITEXOpaqueDataPointerFromPjRtBuffer(PJRT_Buffer* pjrt_c_buffer);

PJRT_Buffer* ITEXCreateSEPjRtBuffer(int device_id, std::string data_type,
                                  std::vector<int64_t>* dimentions, size_t size,
                                  PJRT_Client* pjrt_c_client);

PJRT_Buffer* ITEXCreateSEPjRtBuffer(int device_id, std::string data_type,
                                    std::vector<int64_t> dimentions,
                                    std::vector<int64_t> layout,
                                    PJRT_Client* pjrt_c_client);



#endif  // XLA_PJRT_TF_PJRT_HELPER_H_
