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
// =============================================================

#ifndef XLA_PROFILER_UTILS_H_
#define XLA_PROFILER_UTILS_H_

#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

#define PTI_ASSERT(X) assert(X)

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define MAX_STR_SIZE 1024

#define BYTES_IN_MBYTES (1024 * 1024)

#define NSEC_IN_USEC 1000
#define MSEC_IN_SEC 1000
#define NSEC_IN_MSEC 1000000
#define NSEC_IN_SEC 1000000000

namespace xla{
namespace profiler {
// Converts among different time units.
// NOTE: We use uint64_t for picoseconds and nanoseconds, which are used in
// storage, and double for other units that are used in the UI.
inline double PicosToNanos(uint64_t ps) { return ps / 1E3; }
inline double PicosToMicros(uint64_t ps) { return ps / 1E6; }
inline double PicosToMillis(uint64_t ps) { return ps / 1E9; }
inline double PicosToSeconds(uint64_t ps) { return ps / 1E12; }
inline uint64_t NanosToPicos(uint64_t ns) { return ns * 1000; }
inline double NanosToMicros(uint64_t ns) { return ns / 1E3; }
inline double MicrosToNanos(double us) { return us * 1E3; }
inline double MicrosToMillis(double us) { return us / 1E3; }
inline uint64_t MillisToPicos(double ms) { return ms * 1E9; }
inline uint64_t MillisToNanos(double ms) { return ms * 1E6; }
inline double MillisToSeconds(double ms) { return ms / 1E3; }
inline uint64_t SecondsToNanos(double s) { return s * 1E9; }

// Sleeps for the specified duration.
void SleepForNanos(int64_t ns);
inline void SleepForMicros(int64_t us) { SleepForNanos(us * 1000); }
inline void SleepForMillis(int64_t ms) { SleepForNanos(ms * 1000000); }
inline void SleepForSeconds(int64_t s) { SleepForNanos(s * 1000000000); }

// Spins to simulate doing some work instead of sleeping, because sleep
// precision is poor. For testing only.
void SpinForNanos(int64_t ns);
inline void SpinForMicros(int64_t us) { SpinForNanos(us * 1000); }

} // namespace profiler
} // namespace xla

namespace utils {

static std::atomic<int> g_immediate_command_list_enabled(1);

inline bool IsImmediateCommandListEnabled() {
  return g_immediate_command_list_enabled.load(std::memory_order_acquire);
}

inline void ImmediateCommandListDisabled() {
  g_immediate_command_list_enabled.store(0, std::memory_order_release);
}

inline void SetEnv(const char* name, const char* value) {
  PTI_ASSERT(name != nullptr);
  PTI_ASSERT(value != nullptr);

  int status = 0;
#if defined(_WIN32)
  std::string str = std::string(name) + "=" + value;
  status = _putenv(str.c_str());
#else
  status = setenv(name, value, 1);
#endif
  PTI_ASSERT(status == 0);
}

inline std::string GetEnv(const char* name) {
  PTI_ASSERT(name != nullptr);
#if defined(_WIN32)
  char* value = nullptr;
  errno_t status = _dupenv_s(&value, nullptr, name);
  PTI_ASSERT(status == 0);
  if (value == nullptr) {
    return std::string();
  }
  std::string result(value);
  free(value);
  return result;
#else
  const char* value = getenv(name);
  if (value == nullptr) {
    return std::string();
  }
  return std::string(value);
#endif
}

inline uint32_t GetPid() {
#if defined(_WIN32)
  return GetCurrentProcessId();
#else
  return getpid();
#endif
}

inline uint32_t GetTid() {
#if defined(_WIN32)
  return GetCurrentThreadId();
#else
#ifdef SYS_gettid
  return syscall(SYS_gettid);
#else
#error "SYS_gettid is unavailable on this system"
#endif
#endif
}

}  // namespace utils

#endif  // XLA_PROFILER_UTILS_H_
