/* Copyright (c) 2021 Intel Corporation

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

//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef XLA_PROFILER_TRACE_OPTIONS_H_
#define XLA_PROFILER_TRACE_OPTIONS_H_

#include <sstream>
#include <string>

#include "xla/profiler/utils.h"

#define TRACE_CALL_LOGGING 0
#define TRACE_HOST_TIMING 1
#define TRACE_DEVICE_TIMING 2
#define TRACE_DEVICE_TIMING_VERBOSE 3
#define TRACE_DEVICE_TIMELINE 4
#define TRACE_CHROME_CALL_LOGGING 5
#define TRACE_CHROME_DEVICE_TIMELINE 6
#define TRACE_CHROME_DEVICE_STAGES 7
#define TRACE_TID 8
#define TRACE_PID 9
#define TRACE_LOG_TO_FILE 10
#define TRACE_HOST_RUNTIME_TIMING 11

class TraceOptions {
 public:
  TraceOptions(uint32_t flags)
      : flags_(flags) {
    if (flags_ == 0) {
      flags_ |= (1 << TRACE_HOST_TIMING);
      flags_ |= (1 << TRACE_DEVICE_TIMING);
    }
  }

  bool CheckFlag(uint32_t flag) const { return (flags_ & (1 << flag)); }

 private:
  uint32_t flags_;
};

#endif  // XLA_PROFILER_TRACE_OPTIONS_H_
