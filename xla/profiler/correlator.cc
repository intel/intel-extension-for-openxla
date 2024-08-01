/* Copyright (c) 2024 Intel Corporation

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

#include "xla/profiler/correlator.h"

thread_local uint64_t Correlator::kernel_id_ = 0;
namespace xla{
namespace profiler {
int64_t GetCurrentTimeNanos() {
  // absl::GetCurrentTimeNanos() is much faster than EnvTime::NowNanos().
  // It is wrapped under xla::profiler::GetCurrentTimeNanos to avoid ODR
  // violation and to allow switching to yet another implementation if required.
  return absl::GetCurrentTimeNanos();
};
}  // namespace profiler
}  // namespace xla
// Returns the current CPU wallclock time in nanoseconds.
