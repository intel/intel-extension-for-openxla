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

#include "xla/service/gpu/ccl_ops.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "xla/service/gpu/utils.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

// TODO: It crashes when using public Eigen::bfloat16, need investigation.
#include <sycl/ext/oneapi/bfloat16.hpp>

#if !ITEX_USE_CCL
namespace xla {
namespace gpu {

using bfloat16 = sycl::ext::oneapi::bfloat16;
using float16 = sycl::half;

// After tuned, we found 8 has highest performance and XeLink bandwidth.
constexpr size_t VecBytes = 8;

namespace {
struct Participant {
  Participant(se::gpu::GpuStreamHandle stream, const void* send, void* recv,
              int rank)
      : stream(stream), send(send), recv(recv), rank(rank) {}
  se::gpu::GpuStreamHandle stream;
  const void* send;
  void* recv;
  int rank;
};

struct AlltoAllParticipant {
  se::gpu::GpuStreamHandle stream;
  std::vector<const void*> send;
  std::vector<void*> recv;
  int rank;
};

struct PermuteParticipant {
  se::gpu::GpuStreamHandle stream;
  const void* send;
  void* recv;
  std::optional<int64_t> send_id;
  std::optional<int64_t> recv_id;
  int rank;
};

template <typename T>
struct Collective {
  Collective() : done(false), ready_to_launch(false) {}
  tsl::mutex mu;
  tsl::condition_variable cv;
  bool ready_to_launch TF_GUARDED_BY(mu);
  bool done TF_GUARDED_BY(mu);
  std::vector<T> participants TF_GUARDED_BY(mu);
  std::vector<sycl::event> begin_events TF_GUARDED_BY(mu);
  std::vector<sycl::event> end_events TF_GUARDED_BY(mu);
};

struct Manager {
  static Manager& instance() {
    static Manager m;
    return m;
  }

  tsl::mutex mu;
  std::unordered_map<std::string, std::shared_ptr<Collective<Participant>>>
      collectives TF_GUARDED_BY(mu);
  std::unordered_map<std::string,
                     std::shared_ptr<Collective<AlltoAllParticipant>>>
      alltoall_collectives TF_GUARDED_BY(mu);
  std::unordered_map<std::string,
                     std::shared_ptr<Collective<PermuteParticipant>>>
      permute_collectives TF_GUARDED_BY(mu);
};

template <typename T, typename Func, bool PartialStore>
struct AllReduceKernel;

template <typename T, typename Func, typename AccT = T>
void allreduce_dpcpp(se::gpu::GpuStreamHandle stream, size_t element_count,
                     std::vector<Participant>& participants, int rank,
                     int reduction_size) {
  constexpr size_t VecSize = VecBytes / sizeof(T);
  size_t vec_count = element_count / VecSize;
  size_t vec_tail_element_count = element_count % VecSize;
  size_t total_vec_count = vec_count + (vec_tail_element_count > 0 ? 1 : 0);

  // Each rank allreduces a sub slice of the tensors. Last rank
  // also allreduce the tail vectors of the tensor.
  size_t slice_vec_count = total_vec_count / reduction_size;
  size_t tail_vec_count = total_vec_count % reduction_size;
  size_t local_vec_count =
      slice_vec_count + ((rank == (reduction_size - 1)) ? tail_vec_count : 0);

  if (local_vec_count == 0) return;

  auto device = stream->get_device();
  int group_size =
      device.template get_info<sycl::info::device::max_work_group_size>();

  // set max_workitems = HW_workgroup_num * max_workgroup_size
  int num_max_concurrent_workitem =
      stream_executor::gpu::GpuDriver::GetMultiprocessorCount(&device).value() *
      group_size;
  int num_workitem = local_vec_count <= num_max_concurrent_workitem
                         ? local_vec_count
                         : num_max_concurrent_workitem;
  size_t num_vec_per_workitem = local_vec_count / num_workitem;
  size_t num_tail_vec = local_vec_count % num_workitem;

  int num_workgroup = (num_workitem + group_size - 1) / group_size;

  if (reduction_size <= MAX_RANK_SIZE) {
    stream->submit([&](sycl::handler& cgh) {
      T* in_ptr[MAX_RANK_SIZE];
      T* out_ptr[MAX_RANK_SIZE];

      for (int i = 0; i < reduction_size; ++i) {
        in_ptr[i] = static_cast<T*>(const_cast<void*>(participants[i].send)) +
                    rank * slice_vec_count * VecSize;
        out_ptr[i] = static_cast<T*>(participants[i].recv) +
                     rank * slice_vec_count * VecSize;
      }

      // Last rank may need to process the tail elements which can't form a
      // full vector and need partial block store.
      if (rank != (reduction_size - 1) || vec_tail_element_count == 0) {
        cgh.parallel_for<AllReduceKernel<T, Func, false>>(
            sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                              sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
              const int index = item.get_global_linear_id();
              if (index >= num_workitem) return;

              for (size_t n = 0; n < num_vec_per_workitem; ++n) {
                size_t offset = (num_workitem * n + index) * VecSize;
                AlignedVector<AccT, VecSize, Func> result;
                result.Load(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                    &(in_ptr[0][offset])));
                for (int i = 1; i < reduction_size; ++i)
                  result.Accumulate(
                      *reinterpret_cast<AlignedVector<T, VecSize>*>(
                          &(in_ptr[i][offset])));
                for (int i = 0; i < reduction_size; ++i)
                  result.Store(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                      &(out_ptr[i][offset])));
              }

              if (index < num_tail_vec) {
                size_t offset =
                    (num_workitem * num_vec_per_workitem + index) * VecSize;
                AlignedVector<AccT, VecSize, Func> result;
                result.Load(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                    &(in_ptr[0][offset])));
                for (int i = 1; i < reduction_size; ++i)
                  result.Accumulate(
                      *reinterpret_cast<AlignedVector<T, VecSize>*>(
                          &(in_ptr[i][offset])));
                for (int i = 0; i < reduction_size; ++i)
                  result.Store(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                      &(out_ptr[i][offset])));
              }
            });
      } else {
        cgh.parallel_for<AllReduceKernel<T, Func, true>>(
            sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                              sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
              const int index = item.get_global_linear_id();
              if (index >= num_workitem) return;

              for (size_t n = 0; n < num_vec_per_workitem; ++n) {
                size_t offset = (num_workitem * n + index) * VecSize;
                AlignedVector<AccT, VecSize, Func> result;
                result.Load(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                    &(in_ptr[0][offset])));
                for (int i = 1; i < reduction_size; ++i)
                  result.Accumulate(
                      *reinterpret_cast<AlignedVector<T, VecSize>*>(
                          &(in_ptr[i][offset])));

                if (local_vec_count > num_workitem ||
                    index != (num_workitem - 1)) {
                  for (int i = 0; i < reduction_size; ++i)
                    result.Store(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                        &(out_ptr[i][offset])));
                } else {  // the last workitem may process a partial vector
                  for (int i = 0; i < reduction_size; ++i)
                    result.PartialStore(
                        *reinterpret_cast<AlignedVector<T, VecSize>*>(
                            &(out_ptr[i][offset])),
                        vec_tail_element_count);
                }
              }

              if (index < num_tail_vec) {
                size_t offset =
                    (num_workitem * num_vec_per_workitem + index) * VecSize;
                AlignedVector<AccT, VecSize, Func> result;
                result.Load(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                    &(in_ptr[0][offset])));
                for (int i = 1; i < reduction_size; ++i)
                  result.Accumulate(
                      *reinterpret_cast<AlignedVector<T, VecSize>*>(
                          &(in_ptr[i][offset])));

                if (index != num_tail_vec - 1) {
                  for (int i = 0; i < reduction_size; ++i)
                    result.Store(*reinterpret_cast<AlignedVector<T, VecSize>*>(
                        &(out_ptr[i][offset])));
                } else {  // the last workitem may process a partial vector
                  for (int i = 0; i < reduction_size; ++i)
                    result.PartialStore(
                        *reinterpret_cast<AlignedVector<T, VecSize>*>(
                            &(out_ptr[i][offset])),
                        vec_tail_element_count);
                }
              }
            });
      }
    });
  } else {
    LOG(FATAL) << "Reduction size " << reduction_size
               << " is not supported in AllReduce.";
  }
}

template <typename T>
struct AllGatherKernel;

template <typename T>
void allgather_dpcpp(se::gpu::GpuStreamHandle stream, int tensor_size,
                     std::vector<Participant>& participants,
                     int reduction_size) {
  if (reduction_size <= MAX_RANK_SIZE) {
    const T* in_ptr[MAX_RANK_SIZE];
    T* out_ptr[MAX_RANK_SIZE];

    for (int i = 0; i < reduction_size; ++i) {
      in_ptr[i] = static_cast<const T*>(participants[i].send);
      out_ptr[i] = static_cast<T*>(participants[i].recv);
    }

    for (int i = 0; i < reduction_size; ++i) {
      stream->memcpy(out_ptr[0] + tensor_size * i, (const void*)in_ptr[i],
                     tensor_size * sizeof(T));
    }

    for (int i = 1; i < reduction_size; ++i) {
      stream->memcpy(out_ptr[i], (void*)out_ptr[0],
                     reduction_size * tensor_size * sizeof(T));
    }
  } else {
    LOG(FATAL) << "Reduction size " << reduction_size
               << " is not supported in AllGather.";
  }
}

template <typename T>
struct AllToAllKernel;

template <typename T>
void alltoall_dpcpp(se::gpu::GpuStreamHandle stream, size_t element_count,
                    std::vector<AlltoAllParticipant>& participants, int rank,
                    int reduction_size) {
  constexpr size_t VecSize = VecBytes / sizeof(T);
  size_t slice_element_count = element_count;
  size_t slice_vec_count = slice_element_count / VecSize;
  size_t slice_vec_tail_element_count = slice_element_count % VecSize;
  size_t slice_total_vec_count =
      slice_vec_count + (slice_vec_tail_element_count > 0 ? 1 : 0);
  size_t total_vec_count = slice_total_vec_count * reduction_size;

  auto device = stream->get_device();
  int group_size =
      device.template get_info<sycl::info::device::max_work_group_size>();

  // set max_workitems = HW_workgroup_num * max_workgroup_size
  int num_max_concurrent_workitem =
      stream_executor::gpu::GpuDriver::GetMultiprocessorCount(&device).value() *
      group_size;
  int num_workitem = total_vec_count <= num_max_concurrent_workitem
                         ? total_vec_count
                         : num_max_concurrent_workitem;
  size_t num_vec_per_workitem = total_vec_count / num_workitem;
  size_t num_tail_vec = total_vec_count % num_workitem;

  int num_workgroup = (num_workitem + group_size - 1) / group_size;

  // Process: send vec -> rev vec
  // P0: (a0, a1) -> (a0, b0)
  // P1: (b0, b1) -> (a1, b1)
  if (reduction_size <= MAX_RANK_SIZE) {
    stream->submit([&](sycl::handler& cgh) {
      T* send[MAX_RANK_SIZE];
      T* recv[MAX_RANK_SIZE];

      // Each rank sends its send_buffers to all other ranks.
      for (int i = 0; i < reduction_size; ++i) {
        send[i] =
            const_cast<T*>(static_cast<const T*>(participants[rank].send[i]));
        recv[i] = static_cast<T*>(participants[i].recv[rank]);
      }

      cgh.parallel_for<AllToAllKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          [=](sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= num_workitem) return;

            for (size_t n = 0; n < num_vec_per_workitem; ++n) {
              size_t vec_id = n * num_workitem + index;
              size_t slice_id = vec_id / slice_total_vec_count;
              size_t slice_vec_offset = vec_id % slice_total_vec_count;

              T* send_ptr = send[slice_id] + slice_vec_offset * VecSize;
              T* recv_ptr = recv[slice_id] + slice_vec_offset * VecSize;
              if (slice_vec_tail_element_count == 0 ||
                  slice_vec_offset != slice_vec_count) {
                AlignedVector<T, VecSize> result;
                result.Load(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(send_ptr));
                result.Store(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(recv_ptr));
              } else {
                AlignedVector<T, VecSize> result;
                result.Load(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(send_ptr));
                result.PartialStore(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(recv_ptr),
                    slice_vec_tail_element_count);
              }
            }

            if (index < num_tail_vec) {
              size_t vec_id = num_vec_per_workitem * num_workitem + index;
              size_t slice_id = vec_id / slice_total_vec_count;
              size_t slice_vec_offset = vec_id % slice_total_vec_count;

              T* send_ptr = send[slice_id] + slice_vec_offset * VecSize;
              T* recv_ptr = recv[slice_id] + slice_vec_offset * VecSize;
              if (slice_vec_tail_element_count == 0 ||
                  slice_vec_offset != slice_vec_count) {
                AlignedVector<T, VecSize> result;
                result.Load(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(send_ptr));
                result.Store(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(recv_ptr));
              } else {
                AlignedVector<T, VecSize> result;
                result.Load(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(send_ptr));
                result.PartialStore(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(recv_ptr),
                    slice_vec_tail_element_count);
              }
            }
          });
    });
  } else {
    LOG(FATAL) << "Reduction size " << reduction_size
               << " is not supported in AllToAll.";
  }
}

template <typename T>
struct AllToAllSplitKernel;

template <typename T>
void alltoall_split_dpcpp(se::gpu::GpuStreamHandle stream, size_t element_count,
                          std::vector<AlltoAllParticipant>& participants,
                          int rank, int reduction_size) {
  constexpr size_t VecSize = VecBytes / sizeof(T);
  size_t slice_element_count = element_count / reduction_size;
  size_t slice_vec_count = slice_element_count / VecSize;
  size_t slice_vec_tail_element_count = slice_element_count % VecSize;
  size_t slice_total_vec_count =
      slice_vec_count + (slice_vec_tail_element_count > 0 ? 1 : 0);
  size_t total_vec_count = slice_total_vec_count * reduction_size;

  auto device = stream->get_device();
  int group_size =
      device.template get_info<sycl::info::device::max_work_group_size>();

  // set max_workitems = HW_workgroup_num * max_workgroup_size
  int num_max_concurrent_workitem =
      stream_executor::gpu::GpuDriver::GetMultiprocessorCount(&device).value() *
      group_size;
  int num_workitem = total_vec_count <= num_max_concurrent_workitem
                         ? total_vec_count
                         : num_max_concurrent_workitem;
  size_t num_vec_per_workitem = total_vec_count / num_workitem;
  size_t num_tail_vec = total_vec_count % num_workitem;

  int num_workgroup = (num_workitem + group_size - 1) / group_size;

  // Process: send vec -> rev vec
  // P0: ([a0, a1, a2], [a3, a4, a5]) -> ([a0, a1, a2], [b0, b1, b2])
  // P1: ([b0, b1, b2], [b3, b4, b5]) -> ([a3, a4, a5], [b3, b4, b5])
  if (reduction_size <= MAX_RANK_SIZE) {
    stream->submit([&](sycl::handler& cgh) {
      T* send[MAX_RANK_SIZE];
      T* recv[MAX_RANK_SIZE];

      // Buffer size is always 1 in split AllToAll.
      // Each rank sends its send_buffers to all other ranks.
      for (int i = 0; i < reduction_size; ++i) {
        send[i] =
            const_cast<T*>(static_cast<const T*>(participants[rank].send[0])) +
            i * slice_element_count;
        recv[i] = static_cast<T*>(participants[i].recv[0]) +
                  rank * slice_element_count;
      }

      cgh.parallel_for<AllToAllSplitKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          [=](sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= num_workitem) return;

            for (size_t n = 0; n < num_vec_per_workitem; ++n) {
              size_t vec_id = n * num_workitem + index;
              size_t slice_id = vec_id / slice_total_vec_count;
              size_t slice_vec_offset = vec_id % slice_total_vec_count;

              T* send_ptr = send[slice_id] + slice_vec_offset * VecSize;
              T* recv_ptr = recv[slice_id] + slice_vec_offset * VecSize;
              if (slice_vec_tail_element_count == 0 ||
                  slice_vec_offset != slice_vec_count) {
                AlignedVector<T, VecSize> result;
                result.Load(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(send_ptr));
                result.Store(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(recv_ptr));
              } else {
                AlignedVector<T, VecSize> result;
                result.Load(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(send_ptr));
                result.PartialStore(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(recv_ptr),
                    slice_vec_tail_element_count);
              }
            }

            if (index < num_tail_vec) {
              size_t vec_id = num_vec_per_workitem * num_workitem + index;
              size_t slice_id = vec_id / slice_total_vec_count;
              size_t slice_vec_offset = vec_id % slice_total_vec_count;

              T* send_ptr = send[slice_id] + slice_vec_offset * VecSize;
              T* recv_ptr = recv[slice_id] + slice_vec_offset * VecSize;
              if (slice_vec_tail_element_count == 0 ||
                  slice_vec_offset != slice_vec_count) {
                AlignedVector<T, VecSize> result;
                result.Load(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(send_ptr));
                result.Store(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(recv_ptr));
              } else {
                AlignedVector<T, VecSize> result;
                result.Load(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(send_ptr));
                result.PartialStore(
                    *reinterpret_cast<AlignedVector<T, VecSize>*>(recv_ptr),
                    slice_vec_tail_element_count);
              }
            }
          });
    });
  } else {
    LOG(FATAL) << "Reduction size " << reduction_size
               << " is not supported in AllToAll.";
  }
}

template <typename T, typename Func>
struct ReduceScatterKernel;

template <typename T, typename Func, typename AccT = T>
void reducescatter_dpcpp(se::gpu::GpuStreamHandle stream, int tensor_size,
                         std::vector<Participant>& participants,
                         int reduction_size) {
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  // tensor_size: output tensor size
  auto num_workgroup = (tensor_size + group_size - 1) / group_size;

  if (reduction_size <= MAX_RANK_SIZE) {
    stream->submit([&](sycl::handler& cgh) {
      const T* in[MAX_RANK_SIZE];
      T* out[MAX_RANK_SIZE];

      for (int i = 0; i < reduction_size; ++i) {
        in[i] = static_cast<const T*>(participants[i].send);
        out[i] = static_cast<T*>(participants[i].recv);
      }

      cgh.parallel_for<ReduceScatterKernel<T, Func>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          [=](sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            for (int i = 0; i < reduction_size; ++i) {
              out[i][index] = T(Func()(AccT(in[0][index + tensor_size * i]),
                                       AccT(in[1][index + tensor_size * i])));
              for (int j = 2; j < reduction_size; ++j) {
                out[i][index] = T(Func()(AccT(out[i][index]),
                                         AccT(in[j][index + tensor_size * i])));
              }
            }
          });
    });
  } else {
    LOG(FATAL) << "Reduction size " << reduction_size
               << " is not supported in ReduceScatter.";
  }
}

template <typename T, int size>
struct CollectivePermuteKernel;

template <typename T>
void permute_dpcpp(se::gpu::GpuStreamHandle stream, int tensor_size,
                   std::vector<PermuteParticipant>& participants,
                   int reduction_size) {
  if (reduction_size <= MAX_RANK_SIZE) {
    for (int i = 0; i < reduction_size; ++i)
      if (participants[i].send_id)
        stream->memcpy(participants[i].recv,
                       (const void*)participants[*participants[i].send_id].send,
                       tensor_size * sizeof(T));

  } else {
    LOG(FATAL) << "Reduction size " << reduction_size
               << " is not supported in Permute.";
  }
}

template <class T>
void stream_wait_streamlist(se::gpu::GpuStreamHandle stream,
                            const std::vector<T>& p) {
  std::vector<sycl::event> event_list;
  for (int i = 1; i < p.size(); i++) {
    sycl::event event = SYCLGetEventFromStream(p[i].stream);
    event_list.push_back(event);
  }
  SYCLStreamDependOnEvents(stream, event_list);
}

template <class T>
void streamlist_wait_stream(se::gpu::GpuStreamHandle stream,
                            const std::vector<T>& p) {
  sycl::event event = SYCLGetEventFromStream(stream);

  const std::vector<sycl::event> event_list{event};
  for (int i = 1; i < p.size(); i++) {
    SYCLStreamDependOnEvents(p[i].stream, event_list);
  }
}
}  // namespace

void sycl_allreduce(const void* send_buffer, void* recv_buffer,
                    size_t element_count, PrimitiveType dtype,
                    ReductionKind reduction_kind,
                    se::gpu::GpuStreamHandle gpu_stream, ncclComm_t comm) {
  gpu_stream
      ->wait();  // TODO(intel):remove this wait once barrier bug is fixed.
  std::shared_ptr<Collective<Participant>> collective;
  {
    tsl::mutex_lock l(Manager::instance().mu);
    if (Manager::instance().collectives.find(comm->id) ==
        Manager::instance().collectives.end()) {
      collective = std::make_shared<Collective<Participant>>();
      collective->participants.push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      collective->begin_events.push_back(SYCLGetEventFromStream(gpu_stream));
      Manager::instance().collectives[comm->id] = collective;
    } else {
      collective = Manager::instance().collectives[comm->id];
      tsl::mutex_lock lock(collective->mu);
      collective->participants.push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      collective->begin_events.push_back(SYCLGetEventFromStream(gpu_stream));

      if (collective->participants.size() == comm->nranks) {
        Manager::instance().collectives.erase(comm->id);
        auto& p = collective->participants;
        std::sort(p.begin(), p.end(),
                  [](const Participant& a, const Participant& b) -> bool {
                    return a.rank < b.rank;
                  });
        collective->ready_to_launch = true;
        collective->cv.notify_all();
      }
    }
  }

  {
    tsl::mutex_lock lock(collective->mu);
    if (!collective->ready_to_launch) {
      collective->cv.wait(lock);
    }
  }

  auto& p = collective->participants;
  // TODO(intel):uncomment this barrier once barrier bug is fixed.
  // SYCLStreamDependOnEvents(gpu_stream, collective->begin_events);

  if (reduction_kind == ReductionKind::SUM) {
    if (dtype == PRED)
      allreduce_dpcpp<bool, sycl::plus<bool>>(gpu_stream, element_count, p,
                                              comm->rank, comm->nranks);
    else if (dtype == F32 || dtype == C64)
      allreduce_dpcpp<float, sycl::plus<float>>(gpu_stream, element_count, p,
                                                comm->rank, comm->nranks);
    else if (dtype == F64 || dtype == C128)
      allreduce_dpcpp<double, sycl::plus<double>>(gpu_stream, element_count, p,
                                                  comm->rank, comm->nranks);
    else if (dtype == S32)
      allreduce_dpcpp<int32_t, sycl::plus<int32_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == S64)
      allreduce_dpcpp<int64_t, sycl::plus<int64_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == U32)
      allreduce_dpcpp<uint32_t, sycl::plus<uint32_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == U64)
      allreduce_dpcpp<uint64_t, sycl::plus<uint64_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == BF16)
      allreduce_dpcpp<bfloat16, sycl::plus<float>, float>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else
      LOG(FATAL) << "PrimitiveType "
                 << primitive_util::LowercasePrimitiveTypeName(dtype)
                 << " is not supported in AllReduce.";
  } else if (reduction_kind == ReductionKind::PRODUCT) {
    if (dtype == PRED)
      allreduce_dpcpp<bool, sycl::multiplies<bool>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == F32 || dtype == C64)
      allreduce_dpcpp<float, sycl::multiplies<float>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == F64 || dtype == C128)
      allreduce_dpcpp<double, sycl::multiplies<double>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == S32)
      allreduce_dpcpp<int32_t, sycl::multiplies<int32_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == S64)
      allreduce_dpcpp<int64_t, sycl::multiplies<int64_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == U32)
      allreduce_dpcpp<uint32_t, sycl::multiplies<uint32_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == U64)
      allreduce_dpcpp<uint64_t, sycl::multiplies<uint64_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == BF16)
      allreduce_dpcpp<bfloat16, sycl::multiplies<float>, float>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else
      LOG(FATAL) << "PrimitiveType "
                 << primitive_util::LowercasePrimitiveTypeName(dtype)
                 << " is not supported in AllReduce.";
  } else if (reduction_kind == ReductionKind::MIN) {
    if (dtype == PRED)
      allreduce_dpcpp<bool, sycl::minimum<bool>>(gpu_stream, element_count, p,
                                                 comm->rank, comm->nranks);
    else if (dtype == F32)
      allreduce_dpcpp<float, sycl::minimum<float>>(gpu_stream, element_count, p,
                                                   comm->rank, comm->nranks);
    else if (dtype == F64)
      allreduce_dpcpp<double, sycl::minimum<double>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == S32)
      allreduce_dpcpp<int32_t, sycl::minimum<int32_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == S64)
      allreduce_dpcpp<int64_t, sycl::minimum<int64_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == U32)
      allreduce_dpcpp<uint32_t, sycl::minimum<uint32_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == U64)
      allreduce_dpcpp<uint64_t, sycl::minimum<uint64_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == BF16)
      allreduce_dpcpp<bfloat16, sycl::minimum<float>, float>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else
      LOG(FATAL) << "PrimitiveType "
                 << primitive_util::LowercasePrimitiveTypeName(dtype)
                 << " is not supported in AllReduce.";
  } else if (reduction_kind == ReductionKind::MAX) {
    if (dtype == PRED)
      allreduce_dpcpp<bool, sycl::maximum<bool>>(gpu_stream, element_count, p,
                                                 comm->rank, comm->nranks);
    else if (dtype == F32)
      allreduce_dpcpp<float, sycl::maximum<float>>(gpu_stream, element_count, p,
                                                   comm->rank, comm->nranks);
    else if (dtype == F64)
      allreduce_dpcpp<double, sycl::maximum<double>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == S32)
      allreduce_dpcpp<int32_t, sycl::maximum<int32_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == S64)
      allreduce_dpcpp<int64_t, sycl::maximum<int64_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == U32)
      allreduce_dpcpp<uint32_t, sycl::maximum<uint32_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == U64)
      allreduce_dpcpp<uint64_t, sycl::maximum<uint64_t>>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else if (dtype == BF16)
      allreduce_dpcpp<bfloat16, sycl::maximum<float>, float>(
          gpu_stream, element_count, p, comm->rank, comm->nranks);
    else
      LOG(FATAL) << "PrimitiveType "
                 << primitive_util::LowercasePrimitiveTypeName(dtype)
                 << " is not supported in AllReduce.";
  } else {
    LOG(FATAL) << "ReductionKind " << static_cast<int>(reduction_kind)
               << " is not supported in AllReduce.";
  }

  gpu_stream
      ->wait();  // TODO(intel):remove this wait once barrier bug is fixed.
  {
    tsl::mutex_lock lock(collective->mu);
    collective->end_events.push_back(SYCLGetEventFromStream(gpu_stream));
    if (collective->end_events.size() == comm->nranks) {
      collective->done = true;
      collective->cv.notify_all();
    }

    if (!collective->done) {
      collective->cv.wait(lock);
    }
  }
  // TODO(intel):uncomment this barrier once barrier bug is fixed.
  // SYCLStreamDependOnEvents(gpu_stream, collective->end_events);
}

void sycl_allgather(const void* send_buffer, void* recv_buffer,
                    size_t element_count, PrimitiveType dtype,
                    se::gpu::GpuStreamHandle gpu_stream, ncclComm_t comm) {
  std::shared_ptr<Collective<Participant>> collective;
  bool rank_to_launch_kernel = false;
  {
    tsl::mutex_lock l(Manager::instance().mu);
    if (Manager::instance().collectives.find(comm->id) ==
        Manager::instance().collectives.end()) {
      collective = std::make_shared<Collective<Participant>>();
      collective->participants.push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      Manager::instance().collectives[comm->id] = collective;
    } else {
      collective = Manager::instance().collectives[comm->id];
      tsl::mutex_lock lock(collective->mu);
      collective->participants.push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      if (collective->participants.size() == comm->nranks) {
        Manager::instance().collectives.erase(comm->id);
        rank_to_launch_kernel = true;
      }
    }
  }

  {
    tsl::mutex_lock lock(collective->mu);
    if (!rank_to_launch_kernel) {
      if (!collective->done) collective->cv.wait(lock);
    } else {
      auto p = collective->participants;
      std::sort(p.begin(), p.end(),
                [](const Participant& a, const Participant& b) -> bool {
                  return a.rank < b.rank;
                });

      se::gpu::GpuStreamHandle stream = p[0].stream;
      stream_wait_streamlist(stream, p);

      if (dtype == PRED)
        allgather_dpcpp<bool>(stream, element_count, p, comm->nranks);
      else if (dtype == F32 || dtype == C64)
        allgather_dpcpp<float>(stream, element_count, p, comm->nranks);
      else if (dtype == F64 || dtype == C128)
        allgather_dpcpp<double>(stream, element_count, p, comm->nranks);
      else if (dtype == S32)
        allgather_dpcpp<int32_t>(stream, element_count, p, comm->nranks);
      else if (dtype == S64)
        allgather_dpcpp<int64_t>(stream, element_count, p, comm->nranks);
      else if (dtype == BF16)
        allgather_dpcpp<bfloat16>(stream, element_count, p, comm->nranks);
      else if (dtype == U32)
        allgather_dpcpp<uint32_t>(stream, element_count, p, comm->nranks);
      else if (dtype == U64)
        allgather_dpcpp<uint64_t>(stream, element_count, p, comm->nranks);
      else
        LOG(FATAL) << "PrimitiveType "
                   << primitive_util::LowercasePrimitiveTypeName(dtype)
                   << " is not supported in AllGather.";

      streamlist_wait_stream(stream, p);
      collective->done = true;
      collective->cv.notify_all();
    }
  }
}

void sycl_alltoall(std::vector<const void*> send_buffers,
                   std::vector<void*> recv_buffers, size_t element_count,
                   PrimitiveType dtype, se::gpu::GpuStreamHandle gpu_stream,
                   ncclComm_t comm) {
  gpu_stream
      ->wait();  // TODO(intel):remove this wait once barrier bug is fixed.
  std::shared_ptr<Collective<AlltoAllParticipant>> collective;
  {
    tsl::mutex_lock l(Manager::instance().mu);
    if (Manager::instance().alltoall_collectives.find(comm->id) ==
        Manager::instance().alltoall_collectives.end()) {
      collective = std::make_shared<Collective<AlltoAllParticipant>>();
      collective->participants.push_back(
          {gpu_stream, send_buffers, recv_buffers, comm->rank});
      collective->begin_events.push_back(SYCLGetEventFromStream(gpu_stream));
      Manager::instance().alltoall_collectives[comm->id] = collective;
    } else {
      collective = Manager::instance().alltoall_collectives[comm->id];
      tsl::mutex_lock lock(collective->mu);
      collective->participants.push_back(
          {gpu_stream, send_buffers, recv_buffers, comm->rank});
      collective->begin_events.push_back(SYCLGetEventFromStream(gpu_stream));
      if (collective->participants.size() == comm->nranks) {
        Manager::instance().alltoall_collectives.erase(comm->id);
        auto& p = collective->participants;
        std::sort(p.begin(), p.end(),
                  [](const AlltoAllParticipant& a, const AlltoAllParticipant& b)
                      -> bool { return a.rank < b.rank; });
        collective->ready_to_launch = true;
        collective->cv.notify_all();
      }
    }
  }

  {
    tsl::mutex_lock lock(collective->mu);
    if (!collective->ready_to_launch) {
      collective->cv.wait(lock);
    }
  }

  auto& p = collective->participants;
  // TODO(intel):uncomment this barrier once barrier bug is fixed.
  // SYCLStreamDependOnEvents(gpu_stream, collective->begin_events);

  if (dtype == PRED)
    alltoall_dpcpp<bool>(gpu_stream, element_count, p, comm->rank,
                         comm->nranks);
  else if (dtype == BF16)
    alltoall_dpcpp<bfloat16>(gpu_stream, element_count, p, comm->rank,
                             comm->nranks);
  else if (dtype == F16)
    alltoall_dpcpp<float16>(gpu_stream, element_count, p, comm->rank,
                            comm->nranks);
  else if (dtype == F32 || dtype == C64)
    alltoall_dpcpp<float>(gpu_stream, element_count, p, comm->rank,
                          comm->nranks);
  else if (dtype == F64 || dtype == C128)
    alltoall_dpcpp<double>(gpu_stream, element_count, p, comm->rank,
                           comm->nranks);
  else if (dtype == S8)
    alltoall_dpcpp<int8_t>(gpu_stream, element_count, p, comm->rank,
                           comm->nranks);
  else if (dtype == S16)
    alltoall_dpcpp<int16_t>(gpu_stream, element_count, p, comm->rank,
                            comm->nranks);
  else if (dtype == S32)
    alltoall_dpcpp<int32_t>(gpu_stream, element_count, p, comm->rank,
                            comm->nranks);
  else if (dtype == S64)
    alltoall_dpcpp<int64_t>(gpu_stream, element_count, p, comm->rank,
                            comm->nranks);
  else if (dtype == U8)
    alltoall_dpcpp<uint8_t>(gpu_stream, element_count, p, comm->rank,
                            comm->nranks);
  else if (dtype == U16)
    alltoall_dpcpp<uint16_t>(gpu_stream, element_count, p, comm->rank,
                             comm->nranks);
  else if (dtype == U32)
    alltoall_dpcpp<uint32_t>(gpu_stream, element_count, p, comm->rank,
                             comm->nranks);
  else if (dtype == U64)
    alltoall_dpcpp<uint64_t>(gpu_stream, element_count, p, comm->rank,
                             comm->nranks);
  else
    LOG(FATAL) << "PrimitiveType "
               << primitive_util::LowercasePrimitiveTypeName(dtype)
               << " is not supported in AllToAll.";
  gpu_stream
      ->wait();  // TODO(intel):remove this wait once barrier bug is fixed.
  {
    tsl::mutex_lock lock(collective->mu);
    collective->end_events.push_back(SYCLGetEventFromStream(gpu_stream));
    if (collective->end_events.size() == comm->nranks) {
      collective->done = true;
      collective->cv.notify_all();
    }

    if (!collective->done) {
      collective->cv.wait(lock);
    }
  }
  // TODO(intel):uncomment this barrier once barrier bug is fixed.
  // SYCLStreamDependOnEvents(gpu_stream, collective->end_events);
}

void sycl_alltoall_split(std::vector<const void*> send_buffers,
                         std::vector<void*> recv_buffers, size_t element_count,
                         PrimitiveType dtype,
                         se::gpu::GpuStreamHandle gpu_stream, ncclComm_t comm) {
  gpu_stream
      ->wait();  // TODO(intel):remove this wait once barrier bug is fixed.
  std::shared_ptr<Collective<AlltoAllParticipant>> collective;
  {
    tsl::mutex_lock l(Manager::instance().mu);
    if (Manager::instance().alltoall_collectives.find(comm->id) ==
        Manager::instance().alltoall_collectives.end()) {
      collective = std::make_shared<Collective<AlltoAllParticipant>>();
      collective->participants.push_back(
          {gpu_stream, send_buffers, recv_buffers, comm->rank});
      collective->begin_events.push_back(SYCLGetEventFromStream(gpu_stream));
      Manager::instance().alltoall_collectives[comm->id] = collective;
    } else {
      collective = Manager::instance().alltoall_collectives[comm->id];
      tsl::mutex_lock lock(collective->mu);
      collective->participants.push_back(
          {gpu_stream, send_buffers, recv_buffers, comm->rank});
      collective->begin_events.push_back(SYCLGetEventFromStream(gpu_stream));

      if (collective->participants.size() == comm->nranks) {
        Manager::instance().alltoall_collectives.erase(comm->id);
        auto& p = collective->participants;
        std::sort(p.begin(), p.end(),
                  [](const AlltoAllParticipant& a, const AlltoAllParticipant& b)
                      -> bool { return a.rank < b.rank; });
        collective->ready_to_launch = true;
        collective->cv.notify_all();
      }
    }
  }

  {
    tsl::mutex_lock lock(collective->mu);
    if (!collective->ready_to_launch) {
      collective->cv.wait(lock);
    }
  }

  auto& p = collective->participants;
  // TODO(intel):uncomment this barrier once barrier bug is fixed.
  // SYCLStreamDependOnEvents(gpu_stream, collective->begin_events);

  if (dtype == PRED)
    alltoall_split_dpcpp<bool>(gpu_stream, element_count, p, comm->rank,
                               comm->nranks);
  else if (dtype == BF16)
    alltoall_split_dpcpp<bfloat16>(gpu_stream, element_count, p, comm->rank,
                                   comm->nranks);
  else if (dtype == F16)
    alltoall_split_dpcpp<float16>(gpu_stream, element_count, p, comm->rank,
                                  comm->nranks);
  else if (dtype == F32 || dtype == C64)
    alltoall_split_dpcpp<float>(gpu_stream, element_count, p, comm->rank,
                                comm->nranks);
  else if (dtype == F64 || dtype == C128)
    alltoall_split_dpcpp<double>(gpu_stream, element_count, p, comm->rank,
                                 comm->nranks);
  else if (dtype == S8)
    alltoall_split_dpcpp<int8_t>(gpu_stream, element_count, p, comm->rank,
                                 comm->nranks);
  else if (dtype == S16)
    alltoall_split_dpcpp<int16_t>(gpu_stream, element_count, p, comm->rank,
                                  comm->nranks);
  else if (dtype == S32)
    alltoall_split_dpcpp<int32_t>(gpu_stream, element_count, p, comm->rank,
                                  comm->nranks);
  else if (dtype == S64)
    alltoall_split_dpcpp<int64_t>(gpu_stream, element_count, p, comm->rank,
                                  comm->nranks);
  else if (dtype == U8)
    alltoall_split_dpcpp<uint8_t>(gpu_stream, element_count, p, comm->rank,
                                  comm->nranks);
  else if (dtype == U16)
    alltoall_split_dpcpp<uint16_t>(gpu_stream, element_count, p, comm->rank,
                                   comm->nranks);
  else if (dtype == U32)
    alltoall_split_dpcpp<uint32_t>(gpu_stream, element_count, p, comm->rank,
                                   comm->nranks);
  else if (dtype == U64)
    alltoall_split_dpcpp<uint64_t>(gpu_stream, element_count, p, comm->rank,
                                   comm->nranks);
  else
    LOG(FATAL) << "PrimitiveType "
               << primitive_util::LowercasePrimitiveTypeName(dtype)
               << " is not supported in AllToAll.";

  gpu_stream
      ->wait();  // TODO(intel):remove this wait once barrier bug is fixed.
  {
    tsl::mutex_lock lock(collective->mu);
    collective->end_events.push_back(SYCLGetEventFromStream(gpu_stream));
    if (collective->end_events.size() == comm->nranks) {
      collective->done = true;
      collective->cv.notify_all();
    }

    if (!collective->done) {
      collective->cv.wait(lock);
    }
  }
  // TODO(intel):uncomment this barrier once barrier bug is fixed.
  // SYCLStreamDependOnEvents(gpu_stream, collective->end_events);
}

void sycl_reduce_scatter(const void* send_buffer, void* recv_buffer,
                         size_t element_count, PrimitiveType dtype,
                         ReductionKind reduction_kind,
                         se::gpu::GpuStreamHandle gpu_stream, ncclComm_t comm) {
  std::shared_ptr<Collective<Participant>> collective;
  bool rank_to_launch_kernel = false;
  {
    tsl::mutex_lock l(Manager::instance().mu);
    if (Manager::instance().collectives.find(comm->id) ==
        Manager::instance().collectives.end()) {
      collective = std::make_shared<Collective<Participant>>();
      collective->participants.push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      Manager::instance().collectives[comm->id] = collective;
    } else {
      collective = Manager::instance().collectives[comm->id];
      tsl::mutex_lock lock(collective->mu);
      collective->participants.push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      if (collective->participants.size() == comm->nranks) {
        Manager::instance().collectives.erase(comm->id);
        rank_to_launch_kernel = true;
      }
    }
  }

  {
    tsl::mutex_lock lock(collective->mu);
    if (!rank_to_launch_kernel) {
      if (!collective->done) collective->cv.wait(lock);
    } else {
      auto p = collective->participants;
      std::sort(p.begin(), p.end(),
                [](const Participant& a, const Participant& b) -> bool {
                  return a.rank < b.rank;
                });

      se::gpu::GpuStreamHandle stream = p[0].stream;
      stream_wait_streamlist(stream, p);

      if (reduction_kind == ReductionKind::SUM) {
        if (dtype == PRED)
          reducescatter_dpcpp<bool, sycl::plus<bool>>(stream, element_count, p,
                                                      comm->nranks);
        else if (dtype == F32 || dtype == C64)
          reducescatter_dpcpp<float, sycl::plus<float>>(stream, element_count,
                                                        p, comm->nranks);
        else if (dtype == F64 || dtype == C128)
          reducescatter_dpcpp<double, sycl::plus<double>>(stream, element_count,
                                                          p, comm->nranks);
        else if (dtype == S32)
          reducescatter_dpcpp<int32_t, sycl::plus<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          reducescatter_dpcpp<int64_t, sycl::plus<int64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U32)
          reducescatter_dpcpp<uint32_t, sycl::plus<uint32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U64)
          reducescatter_dpcpp<uint64_t, sycl::plus<uint64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == BF16)
          reducescatter_dpcpp<bfloat16, sycl::plus<float>, float>(
              stream, element_count, p, comm->nranks);
        else
          LOG(FATAL) << "PrimitiveType "
                     << primitive_util::LowercasePrimitiveTypeName(dtype)
                     << " is not supported in ReduceScatter.";
      } else if (reduction_kind == ReductionKind::PRODUCT) {
        if (dtype == PRED)
          reducescatter_dpcpp<bool, sycl::multiplies<bool>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == F32 || dtype == C64)
          reducescatter_dpcpp<float, sycl::multiplies<float>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == F64 || dtype == C128)
          reducescatter_dpcpp<double, sycl::multiplies<double>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S32)
          reducescatter_dpcpp<int32_t, sycl::multiplies<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          reducescatter_dpcpp<int64_t, sycl::multiplies<int64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U32)
          reducescatter_dpcpp<uint32_t, sycl::multiplies<uint32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U64)
          reducescatter_dpcpp<uint64_t, sycl::multiplies<uint64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == BF16)
          reducescatter_dpcpp<bfloat16, sycl::multiplies<float>, float>(
              stream, element_count, p, comm->nranks);
        else
          LOG(FATAL) << "PrimitiveType "
                     << primitive_util::LowercasePrimitiveTypeName(dtype)
                     << " is not supported in ReduceScatter.";
      } else if (reduction_kind == ReductionKind::MIN) {
        if (dtype == PRED)
          reducescatter_dpcpp<bool, sycl::minimum<bool>>(stream, element_count,
                                                         p, comm->nranks);
        else if (dtype == F32 || dtype == C64)
          reducescatter_dpcpp<float, sycl::minimum<float>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == F64 || dtype == C128)
          reducescatter_dpcpp<double, sycl::minimum<double>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S32)
          reducescatter_dpcpp<int32_t, sycl::minimum<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          reducescatter_dpcpp<int64_t, sycl::minimum<int64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == BF16)
          reducescatter_dpcpp<bfloat16, sycl::minimum<float>, float>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U32)
          reducescatter_dpcpp<uint32_t, sycl::minimum<uint32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U64)
          reducescatter_dpcpp<uint64_t, sycl::minimum<uint64_t>>(
              stream, element_count, p, comm->nranks);
        else
          LOG(FATAL) << "PrimitiveType "
                     << primitive_util::LowercasePrimitiveTypeName(dtype)
                     << " is not supported in ReduceScatter.";
      } else if (reduction_kind == ReductionKind::MAX) {
        if (dtype == PRED)
          reducescatter_dpcpp<bool, sycl::maximum<bool>>(stream, element_count,
                                                         p, comm->nranks);
        else if (dtype == F32 || dtype == C64)
          reducescatter_dpcpp<float, sycl::maximum<float>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == F64 || dtype == C128)
          reducescatter_dpcpp<double, sycl::maximum<double>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S32)
          reducescatter_dpcpp<int32_t, sycl::maximum<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          reducescatter_dpcpp<int64_t, sycl::maximum<int64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == BF16)
          reducescatter_dpcpp<bfloat16, sycl::maximum<float>, float>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U32)
          reducescatter_dpcpp<uint32_t, sycl::maximum<uint32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U64)
          reducescatter_dpcpp<uint64_t, sycl::maximum<uint64_t>>(
              stream, element_count, p, comm->nranks);
        else
          LOG(FATAL) << "PrimitiveType "
                     << primitive_util::LowercasePrimitiveTypeName(dtype)
                     << " is not supported in ReduceScatter.";
      } else {
        LOG(FATAL) << "ReductionKind " << static_cast<int>(reduction_kind)
                   << " is not supported in ReduceScatter.";
      }

      streamlist_wait_stream(stream, p);
      collective->done = true;
      collective->cv.notify_all();
    }
  }
}

void sycl_collective_permute(const void* send_buffer, void* recv_buffer,
                             size_t element_count, PrimitiveType dtype,
                             const std::optional<int64_t>& source_id,
                             const std::optional<int64_t>& target_id,
                             se::gpu::GpuStreamHandle gpu_stream,
                             ncclComm_t comm) {
  std::shared_ptr<Collective<PermuteParticipant>> collective;
  bool rank_to_launch_kernel = false;
  {
    tsl::mutex_lock l(Manager::instance().mu);
    if (Manager::instance().permute_collectives.find(comm->id) ==
        Manager::instance().permute_collectives.end()) {
      collective = std::make_shared<Collective<PermuteParticipant>>();
      collective->participants.push_back({gpu_stream, send_buffer, recv_buffer,
                                          source_id, target_id, comm->rank});
      Manager::instance().permute_collectives[comm->id] = collective;
    } else {
      collective = Manager::instance().permute_collectives[comm->id];
      tsl::mutex_lock lock(collective->mu);
      collective->participants.push_back({gpu_stream, send_buffer, recv_buffer,
                                          source_id, target_id, comm->rank});
      if (collective->participants.size() == comm->nranks) {
        Manager::instance().permute_collectives.erase(comm->id);
        rank_to_launch_kernel = true;
      }
    }
  }

  {
    tsl::mutex_lock lock(collective->mu);
    if (!rank_to_launch_kernel) {
      if (!collective->done) collective->cv.wait(lock);
    } else {
      auto p = collective->participants;
      std::sort(
          p.begin(), p.end(),
          [](const PermuteParticipant& a, const PermuteParticipant& b) -> bool {
            return a.rank < b.rank;
          });

      se::gpu::GpuStreamHandle stream = p[0].stream;
      stream_wait_streamlist(stream, p);

      if (dtype == PRED)
        permute_dpcpp<bool>(stream, element_count, p, comm->nranks);
      else if (dtype == F32)
        permute_dpcpp<float>(stream, element_count, p, comm->nranks);
      else if (dtype == F64)
        permute_dpcpp<double>(stream, element_count, p, comm->nranks);
      else if (dtype == S32)
        permute_dpcpp<int32_t>(stream, element_count, p, comm->nranks);
      else if (dtype == S64)
        permute_dpcpp<int64_t>(stream, element_count, p, comm->nranks);
      else if (dtype == BF16)
        permute_dpcpp<bfloat16>(stream, element_count, p, comm->nranks);
      else if (dtype == U32)
        permute_dpcpp<uint32_t>(stream, element_count, p, comm->nranks);
      else if (dtype == U64)
        permute_dpcpp<uint64_t>(stream, element_count, p, comm->nranks);
      else
        LOG(FATAL) << "PrimitiveType "
                   << primitive_util::LowercasePrimitiveTypeName(dtype)
                   << " is not supported in Permute.";

      streamlist_wait_stream(stream, p);
      collective->done = true;
      collective->cv.notify_all();
    }
  }
}

}  // namespace gpu
}  // namespace xla
#endif  // ITEX_USE_CCL