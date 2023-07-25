/* Copyright (c) 2023 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_CCL_UTILS_H_
#define XLA_SERVICE_GPU_CCL_UTILS_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"

#if ITEX_USE_CCL
#include "oneapi/ccl.hpp"
#else
namespace ccl {
struct communicator {
  communicator(int nranks, int rank, const std::string& id)
      : nranks(nranks), rank(rank), id(id) {}
  int nranks;
  int rank;
  const std::string& id;
};
}  // namespace ccl
#endif  // ITEX_USE_CCL

namespace xla {
namespace gpu {

#if ITEX_USE_CCL
ccl::reduction ToCclReduction(ReductionKind kind);
StatusOr<std::pair<ccl::datatype, int>> ToCclDataTypeAndCountMultiplier(
    PrimitiveType element_type);
#endif  // ITEX_USE_CCL

bool IsGlobalCclConfig();
bool IsCclLaunchModeParallel();

size_t GetNumLocalParticipants(
    const std::vector<GlobalDeviceId>& participants,
    const std::vector<GlobalDeviceId>* local_devices);  // may be null

StatusOr<const NcclUniqueIdCallback*> GetNcclUniqueIdCallback(
    const NcclUniqueIdCallback* unique_id_callback,  // may be null
    bool is_local);

// Represents a type that requires mutually exclusive access.
template <typename T>
class Lockable {
 public:
  // RAII type that will release the exclusive lock when it is destroyed.
  using Lock = std::unique_ptr<T, std::function<void(T*)>>;

  Lockable() = default;
  explicit Lockable(T value) : value_(std::move(value)) {}
  Lockable(const Lockable&) = delete;
  Lockable(Lockable&&) = delete;
  Lockable& operator=(const Lockable&) = delete;
  Lockable& operator=(Lockable&&) = delete;

  Lock Acquire() {
    absl::MutexLock lock(&mutex_);
    mutex_.Await(absl::Condition(&is_unlocked_));
    is_unlocked_ = false;

    return {&value_, [this](T*) {
              absl::MutexLock lock(&mutex_);
              CHECK(!is_unlocked_);
              is_unlocked_ = true;
            }};
  }

 private:
  T value_;
  absl::Mutex mutex_;
  bool is_unlocked_ ABSL_GUARDED_BY(mutex_) = true;
};

TSL_LIB_GTL_DEFINE_INT_TYPE(OpId, int64_t);

struct CclComm : public Lockable<ccl::communicator*> {
  explicit CclComm(ccl::communicator* comm) : Lockable(comm) {}
};

StatusOr<CclComm::Lock> AcquireCclComm(
    RunId run_id, OpId op_id, std::vector<GlobalDeviceId> participants,
    size_t num_local_participants,
    const NcclUniqueIdCallback& unique_id_callback, int rank,
    int64_t stream_id);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CCL_UTILS_H_
