/* Copyright (c) 2021-2022 Intel Corporation

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
//==============================================================
#ifndef XLA_PROFILER_CORRELATOR_H_
#define XLA_PROFILER_CORRELATOR_H_

#include <level_zero/ze_api.h>

#include <map>
#include <vector>
#include <iostream>

#include "absl/time/clock.h"
#include "xla/profiler/utils.h"

namespace xla{
namespace profiler {
// Returns the current CPU wallclock time in nanoseconds.
int64_t GetCurrentTimeNanos();

}  // namespace profiler
}  // namespace xla

struct ApiCollectorOptions {
  bool call_tracing;
  bool need_tid;
  bool need_pid;
};

class Correlator {
 public:
  Correlator() : base_time_(xla::profiler::GetCurrentTimeNanos()) {
  }

  uint64_t GetTimestamp() const {
    return xla::profiler::GetCurrentTimeNanos() - base_time_;
  }

  uint64_t GetStartPoint() const { return base_time_; }

  uint64_t GetKernelIdVector() const { return kernel_id_; }

  void SetKernelId(uint64_t kernel_id) { kernel_id_ = kernel_id; }

  std::vector<uint64_t> GetKernelIdVector (ze_command_list_handle_t command_list) {
    if (kernel_id_map_.count(command_list) > 0) {
      return kernel_id_map_[command_list];
    } else {
      return std::vector<uint64_t>();
    }
  }

  void CreateKernelIdList(ze_command_list_handle_t command_list) {
    kernel_id_map_[command_list] = std::vector<uint64_t>();
  }

  void RemoveKernelIdList(ze_command_list_handle_t command_list) {
    kernel_id_map_.erase(command_list);
  }

  void ResetKernelIdList(ze_command_list_handle_t command_list) {
    kernel_id_map_[command_list].clear();
  }

  void AddKernelId(ze_command_list_handle_t command_list, uint64_t kernel_id) {
    kernel_id_map_[command_list].push_back(kernel_id);
  }

  std::vector<uint64_t> GetCallIdVector(ze_command_list_handle_t command_list) {
    if (call_id_map_.count(command_list) > 0) {
      return call_id_map_[command_list];
    } else {
      return std::vector<uint64_t>();
    }
  }

  void CreateCallIdList(ze_command_list_handle_t command_list) {
    call_id_map_[command_list] = std::vector<uint64_t>();
  }

  void RemoveCallIdList(ze_command_list_handle_t command_list) {
    call_id_map_.erase(command_list);
  }

  void ResetCallIdList(ze_command_list_handle_t command_list) {
    call_id_map_[command_list].clear();
  }

  void AddCallId(ze_command_list_handle_t command_list, uint64_t call_id) {
    call_id_map_[command_list].push_back(call_id);
  }

 private:
  uint64_t base_time_;
  std::map<ze_command_list_handle_t, std::vector<uint64_t> > kernel_id_map_;
  std::map<ze_command_list_handle_t, std::vector<uint64_t> > call_id_map_;

  static thread_local uint64_t kernel_id_;
};

#endif  // XLA_PROFILER_CORRELATOR_H_
