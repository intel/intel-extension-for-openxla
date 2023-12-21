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

#include "xla/service/gpu/ccl_ops.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

// TODO: It crashes when using public Eigen::bfloat16, need investigation.
#include <sycl/ext/oneapi/bfloat16.hpp>

#if !ITEX_USE_CCL
namespace xla {
namespace gpu {

using bfloat16 = sycl::ext::oneapi::bfloat16;

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

struct Manager {
  static Manager& instance() {
    static Manager m;
    return m;
  }

  tsl::mutex mu;
  tsl::condition_variable cv;
  // The order should be: (r0p0, r1p0), (r0p1, r1p1)
  std::unordered_map<std::string, std::vector<Participant>> collectives
      TF_GUARDED_BY(mu);
  std::unordered_map<std::string, std::vector<AlltoAllParticipant>>
      alltoall_collectives TF_GUARDED_BY(mu);
  std::unordered_map<std::string, std::vector<PermuteParticipant>>
      permute_collectives TF_GUARDED_BY(mu);
};

template <typename T, typename Func, int reduction_size>
struct AllReduceKernel;

template <typename T, typename Func, typename AccT = T>
void allreduce_dpcpp(se::gpu::GpuStreamHandle stream, int tensor_size,
                     std::vector<Participant>& participants,
                     int reduction_size) {
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_workgroup = (tensor_size + group_size - 1) / group_size;

#define IN(i) in##i##_ptr
#define OUT(i) out##i##_ptr
#define IN_PTR(i) auto IN(i) = static_cast<const T*>(participants[i].send)
#define OUT_PTR(i) auto OUT(i) = static_cast<T*>(participants[i].recv)
#define ADD_DPCPP(i, j) \
  OUT(i)[index] = T(Func()(AccT(OUT(i)[index]), AccT(IN(j)[index])))
#define COPY_OUT_DPCPP(i, j) OUT(i)[index] = OUT(j)[index]

  if (reduction_size == 2) {
    stream->submit([&](sycl::handler& cgh) {
      IN_PTR(0);
      IN_PTR(1);
      OUT_PTR(0);
      OUT_PTR(1);

      cgh.parallel_for<AllReduceKernel<T, Func, 2>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          [=](sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            ADD_DPCPP(0, 1);
            COPY_OUT_DPCPP(1, 0);
          });
    });
  } else if (reduction_size == 4) {
    stream->submit([&](sycl::handler& cgh) {
      IN_PTR(0);
      IN_PTR(1);
      IN_PTR(2);
      IN_PTR(3);
      OUT_PTR(0);
      OUT_PTR(1);
      OUT_PTR(2);
      OUT_PTR(3);

      cgh.parallel_for<AllReduceKernel<T, Func, 4>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          [=](sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            ADD_DPCPP(0, 1);
            ADD_DPCPP(0, 2);
            ADD_DPCPP(0, 3);
            COPY_OUT_DPCPP(1, 0);
            COPY_OUT_DPCPP(2, 0);
            COPY_OUT_DPCPP(3, 0);
          });
    });
  } else if (reduction_size == 6) {
    stream->submit([&](sycl::handler& cgh) {
      IN_PTR(0);
      IN_PTR(1);
      IN_PTR(2);
      IN_PTR(3);
      IN_PTR(4);
      IN_PTR(5);
      OUT_PTR(0);
      OUT_PTR(1);
      OUT_PTR(2);
      OUT_PTR(3);
      OUT_PTR(4);
      OUT_PTR(5);

      cgh.parallel_for<AllReduceKernel<T, Func, 6>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          [=](sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            ADD_DPCPP(0, 1);
            ADD_DPCPP(0, 2);
            ADD_DPCPP(0, 3);
            ADD_DPCPP(0, 4);
            ADD_DPCPP(0, 5);
            COPY_OUT_DPCPP(1, 0);
            COPY_OUT_DPCPP(2, 0);
            COPY_OUT_DPCPP(3, 0);
            COPY_OUT_DPCPP(4, 0);
            COPY_OUT_DPCPP(5, 0);
          });
    });
  } else if (reduction_size == 8) {
    stream->submit([&](sycl::handler& cgh) {
      IN_PTR(0);
      IN_PTR(1);
      IN_PTR(2);
      IN_PTR(3);
      IN_PTR(4);
      IN_PTR(5);
      IN_PTR(6);
      IN_PTR(7);
      OUT_PTR(0);
      OUT_PTR(1);
      OUT_PTR(2);
      OUT_PTR(3);
      OUT_PTR(4);
      OUT_PTR(5);
      OUT_PTR(6);
      OUT_PTR(7);

      cgh.parallel_for<AllReduceKernel<T, Func, 8>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          [=](sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            ADD_DPCPP(0, 1);
            ADD_DPCPP(0, 2);
            ADD_DPCPP(0, 3);
            ADD_DPCPP(0, 4);
            ADD_DPCPP(0, 5);
            ADD_DPCPP(0, 6);
            ADD_DPCPP(0, 7);
            COPY_OUT_DPCPP(1, 0);
            COPY_OUT_DPCPP(2, 0);
            COPY_OUT_DPCPP(3, 0);
            COPY_OUT_DPCPP(4, 0);
            COPY_OUT_DPCPP(5, 0);
            COPY_OUT_DPCPP(6, 0);
            COPY_OUT_DPCPP(7, 0);
          });
    });
  } else if (reduction_size == 10) {
    stream->submit([&](sycl::handler& cgh) {
      IN_PTR(0);
      IN_PTR(1);
      IN_PTR(2);
      IN_PTR(3);
      IN_PTR(4);
      IN_PTR(5);
      IN_PTR(6);
      IN_PTR(7);
      IN_PTR(8);
      IN_PTR(9);
      OUT_PTR(0);
      OUT_PTR(1);
      OUT_PTR(2);
      OUT_PTR(3);
      OUT_PTR(4);
      OUT_PTR(5);
      OUT_PTR(6);
      OUT_PTR(7);
      OUT_PTR(8);
      OUT_PTR(9);
      ;

      cgh.parallel_for<AllReduceKernel<T, Func, 10>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          [=](sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            ADD_DPCPP(0, 1);
            ADD_DPCPP(0, 2);
            ADD_DPCPP(0, 3);
            ADD_DPCPP(0, 4);
            ADD_DPCPP(0, 5);
            ADD_DPCPP(0, 6);
            ADD_DPCPP(0, 7);
            ADD_DPCPP(0, 8);
            ADD_DPCPP(0, 9);
            COPY_OUT_DPCPP(1, 0);
            COPY_OUT_DPCPP(2, 0);
            COPY_OUT_DPCPP(3, 0);
            COPY_OUT_DPCPP(4, 0);
            COPY_OUT_DPCPP(5, 0);
            COPY_OUT_DPCPP(6, 0);
            COPY_OUT_DPCPP(7, 0);
            COPY_OUT_DPCPP(8, 0);
            COPY_OUT_DPCPP(9, 0);
          });
    });
  } else if (reduction_size == 12) {
    stream->submit([&](sycl::handler& cgh) {
      IN_PTR(0);
      IN_PTR(1);
      IN_PTR(2);
      IN_PTR(3);
      IN_PTR(4);
      IN_PTR(5);
      IN_PTR(6);
      IN_PTR(7);
      IN_PTR(8);
      IN_PTR(9);
      IN_PTR(10);
      IN_PTR(11);
      OUT_PTR(0);
      OUT_PTR(1);
      OUT_PTR(2);
      OUT_PTR(3);
      OUT_PTR(4);
      OUT_PTR(5);
      OUT_PTR(6);
      OUT_PTR(7);
      OUT_PTR(8);
      OUT_PTR(9);
      OUT_PTR(10);
      OUT_PTR(11);

      cgh.parallel_for<AllReduceKernel<T, Func, 12>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          [=](sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;
            ADD_DPCPP(0, 1);
            ADD_DPCPP(0, 2);
            ADD_DPCPP(0, 3);
            ADD_DPCPP(0, 4);
            ADD_DPCPP(0, 5);
            ADD_DPCPP(0, 6);
            ADD_DPCPP(0, 7);
            ADD_DPCPP(0, 8);
            ADD_DPCPP(0, 9);
            ADD_DPCPP(0, 10);
            ADD_DPCPP(0, 11);
            COPY_OUT_DPCPP(1, 0);
            COPY_OUT_DPCPP(2, 0);
            COPY_OUT_DPCPP(3, 0);
            COPY_OUT_DPCPP(4, 0);
            COPY_OUT_DPCPP(5, 0);
            COPY_OUT_DPCPP(6, 0);
            COPY_OUT_DPCPP(7, 0);
            COPY_OUT_DPCPP(8, 0);
            COPY_OUT_DPCPP(9, 0);
            COPY_OUT_DPCPP(10, 0);
            COPY_OUT_DPCPP(11, 0);
          });
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
void alltoall_dpcpp(se::gpu::GpuStreamHandle stream, int tensor_size,
                    std::vector<AlltoAllParticipant>& participants,
                    int reduction_size) {
  const int kLimitedRankSize = MAX_RANK_SIZE / 2;
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  auto num_workgroup = (tensor_size + group_size - 1) / group_size;

  // Process: send vec -> rev vec
  // P0: (a0, a1) -> (a0, b0)
  // P1: (b0, b1) -> (a1, b1)
  if (reduction_size <= kLimitedRankSize) {
    stream->submit([&](sycl::handler& cgh) {
      const T* send[kLimitedRankSize][kLimitedRankSize];
      T* recv[kLimitedRankSize][kLimitedRankSize];

      for (int i = 0; i < reduction_size; ++i) {
        for (int j = 0; j < reduction_size; ++j) {
          send[i][j] = static_cast<const T*>(participants[i].send[j]);
          recv[i][j] = static_cast<T*>(participants[i].recv[j]);
        }
      }

      cgh.parallel_for<AllToAllKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          [=](sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= tensor_size) return;

            for (int i = 0; i < reduction_size; ++i) {
              for (int j = 0; j < reduction_size; ++j) {
                recv[j][i][index] = send[i][j][index];
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
void alltoall_split_dpcpp(se::gpu::GpuStreamHandle stream, int tensor_size,
                          std::vector<AlltoAllParticipant>& participants,
                          int reduction_size) {
  const int kLimitedRankSize = MAX_RANK_SIZE / 2;
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();
  int sub_tensor_size = tensor_size / reduction_size;
  auto num_workgroup = (sub_tensor_size + group_size - 1) / group_size;

  // clang-format off
  // Process: send vec -> rev vec
  // P0: ([a0, a1], [b0, b1], [c0, c1], ...) -> ([a0, d0], [b0, e0], [c0, f0], ...)
  // P1: ([d0, d1], [e0, e1], [f0, f1], ...) -> ([a1, d1], [b1, e1], [c1, f1], ...)
  // clang-format on
  if (reduction_size <= kLimitedRankSize) {
    stream->submit([&](sycl::handler& cgh) {
      const T* send[kLimitedRankSize][kLimitedRankSize];  // SYCL: fix size
      T* recv[kLimitedRankSize][kLimitedRankSize];        // SYCL: fix size

      for (int i = 0; i < reduction_size; ++i) {
        for (int k = 0; k < reduction_size; ++k) {  // SYCL: fix size
          send[i][k] = static_cast<const T*>(participants[i].send[k]);
          recv[i][k] = static_cast<T*>(participants[i].recv[k]);
        }
      }

      cgh.parallel_for<AllToAllSplitKernel<T>>(
          sycl::nd_range<1>(sycl::range<1>(group_size * num_workgroup),
                            sycl::range<1>(group_size)),
          [=](sycl::nd_item<1> item) {
            const int index = item.get_global_linear_id();
            if (index >= sub_tensor_size) return;

            for (int k = 0; k < reduction_size; ++k) {  // SYCL: fix size
              for (int i = 0; i < reduction_size; ++i) {
                for (int j = 0; j < reduction_size; ++j) {
                  recv[i][k][j * sub_tensor_size + index] =
                      send[j][k][i * sub_tensor_size + index];
                }
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
    sycl::event event = p[i].stream->ext_oneapi_submit_barrier();
    event_list.push_back(event);
  }
  stream->ext_oneapi_submit_barrier(event_list);
}

template <class T>
void streamlist_wait_stream(se::gpu::GpuStreamHandle stream,
                            const std::vector<T>& p) {
  sycl::event event = stream->ext_oneapi_submit_barrier();

  const std::vector<sycl::event> event_list{event};
  for (int i = 1; i < p.size(); i++) {
    p[i].stream->ext_oneapi_submit_barrier(event_list);
  }
}
}  // namespace

void sycl_allreduce(const void* send_buffer, void* recv_buffer,
                    int element_count, PrimitiveType dtype,
                    ReductionKind reduction_kind,
                    se::gpu::GpuStreamHandle gpu_stream, ncclComm_t comm,
                    int current_call, int max_call) {
  std::vector<Participant> p;
  {
    tsl::mutex_lock l(Manager::instance().mu);
    if (Manager::instance().collectives.find(comm->id) ==
        Manager::instance().collectives.end()) {
      std::vector<Participant> participants;
      participants.push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      Manager::instance().collectives[comm->id] = participants;
      p = participants;
    } else {
      Manager::instance().collectives[comm->id].push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      p = Manager::instance().collectives[comm->id];
    }

    if (p.size() != comm->nranks) {
      Manager::instance().cv.wait(l);
    } else {
      Manager::instance().collectives.erase(comm->id);
      std::sort(p.begin(), p.end(),
                [](const Participant& a, const Participant& b) -> bool {
                  return a.rank < b.rank;
                });

      se::gpu::GpuStreamHandle stream = p[0].stream;
      if (current_call == 0) stream_wait_streamlist(stream, p);

      if (reduction_kind == ReductionKind::SUM) {
        if (dtype == PRED)
          allreduce_dpcpp<bool, sycl::plus<bool>>(stream, element_count, p,
                                                  comm->nranks);
        else if (dtype == F32)
          allreduce_dpcpp<float, sycl::plus<float>>(stream, element_count, p,
                                                    comm->nranks);
        else if (dtype == F64)
          allreduce_dpcpp<double, sycl::plus<double>>(stream, element_count, p,
                                                      comm->nranks);
        else if (dtype == S32)
          allreduce_dpcpp<int32_t, sycl::plus<int32_t>>(stream, element_count,
                                                        p, comm->nranks);
        else if (dtype == S64)
          allreduce_dpcpp<int64_t, sycl::plus<int64_t>>(stream, element_count,
                                                        p, comm->nranks);
        else if (dtype == U32)
          allreduce_dpcpp<uint32_t, sycl::plus<uint32_t>>(stream, element_count,
                                                          p, comm->nranks);
        else if (dtype == U64)
          allreduce_dpcpp<uint64_t, sycl::plus<uint64_t>>(stream, element_count,
                                                          p, comm->nranks);
        else if (dtype == C64)
          allreduce_dpcpp<std::complex<float>, sycl::plus<std::complex<float>>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == C128)
          allreduce_dpcpp<std::complex<double>,
                          sycl::plus<std::complex<double>>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == BF16)
          allreduce_dpcpp<bfloat16, sycl::plus<float>, float>(
              stream, element_count, p, comm->nranks);
        else
          LOG(FATAL) << "PrimitiveType "
                     << primitive_util::LowercasePrimitiveTypeName(dtype)
                     << " is not supported in AllReduce.";
      } else if (reduction_kind == ReductionKind::PRODUCT) {
        if (dtype == PRED)
          allreduce_dpcpp<bool, sycl::multiplies<bool>>(stream, element_count,
                                                        p, comm->nranks);
        else if (dtype == F32)
          allreduce_dpcpp<float, sycl::multiplies<float>>(stream, element_count,
                                                          p, comm->nranks);
        else if (dtype == F64)
          allreduce_dpcpp<double, sycl::multiplies<double>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S32)
          allreduce_dpcpp<int32_t, sycl::multiplies<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          allreduce_dpcpp<int64_t, sycl::multiplies<int64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U32)
          allreduce_dpcpp<uint32_t, sycl::multiplies<uint32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U64)
          allreduce_dpcpp<uint64_t, sycl::multiplies<uint64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == C64)
          allreduce_dpcpp<std::complex<float>,
                          sycl::multiplies<std::complex<float>>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == C128)
          allreduce_dpcpp<std::complex<double>,
                          sycl::multiplies<std::complex<double>>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == BF16)
          allreduce_dpcpp<bfloat16, sycl::multiplies<float>, float>(
              stream, element_count, p, comm->nranks);
        else
          LOG(FATAL) << "PrimitiveType "
                     << primitive_util::LowercasePrimitiveTypeName(dtype)
                     << " is not supported in AllReduce.";
      } else if (reduction_kind == ReductionKind::MIN) {
        if (dtype == PRED)
          allreduce_dpcpp<bool, sycl::minimum<bool>>(stream, element_count, p,
                                                     comm->nranks);
        else if (dtype == F32)
          allreduce_dpcpp<float, sycl::minimum<float>>(stream, element_count, p,
                                                       comm->nranks);
        else if (dtype == F64)
          allreduce_dpcpp<double, sycl::minimum<double>>(stream, element_count,
                                                         p, comm->nranks);
        else if (dtype == S32)
          allreduce_dpcpp<int32_t, sycl::minimum<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          allreduce_dpcpp<int64_t, sycl::minimum<int64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U32)
          allreduce_dpcpp<uint32_t, sycl::minimum<uint32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U64)
          allreduce_dpcpp<uint64_t, sycl::minimum<uint64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == BF16)
          allreduce_dpcpp<bfloat16, sycl::minimum<float>, float>(
              stream, element_count, p, comm->nranks);
        else
          LOG(FATAL) << "PrimitiveType "
                     << primitive_util::LowercasePrimitiveTypeName(dtype)
                     << " is not supported in AllReduce.";
      } else if (reduction_kind == ReductionKind::MAX) {
        if (dtype == PRED)
          allreduce_dpcpp<bool, sycl::maximum<bool>>(stream, element_count, p,
                                                     comm->nranks);
        else if (dtype == F32)
          allreduce_dpcpp<float, sycl::maximum<float>>(stream, element_count, p,
                                                       comm->nranks);
        else if (dtype == F64)
          allreduce_dpcpp<double, sycl::maximum<double>>(stream, element_count,
                                                         p, comm->nranks);
        else if (dtype == S32)
          allreduce_dpcpp<int32_t, sycl::maximum<int32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == S64)
          allreduce_dpcpp<int64_t, sycl::maximum<int64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U32)
          allreduce_dpcpp<uint32_t, sycl::maximum<uint32_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == U64)
          allreduce_dpcpp<uint64_t, sycl::maximum<uint64_t>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == BF16)
          allreduce_dpcpp<bfloat16, sycl::maximum<float>, float>(
              stream, element_count, p, comm->nranks);
        else
          LOG(FATAL) << "PrimitiveType "
                     << primitive_util::LowercasePrimitiveTypeName(dtype)
                     << " is not supported in AllReduce.";
      } else {
        LOG(FATAL) << "ReductionKind " << static_cast<int>(reduction_kind)
                   << " is not supported in AllReduce.";
      }

      if (current_call == (max_call - 1)) streamlist_wait_stream(stream, p);
      Manager::instance().cv.notify_all();
    }
  }
}

void sycl_allgather(const void* send_buffer, void* recv_buffer,
                    int element_count, PrimitiveType dtype,
                    se::gpu::GpuStreamHandle gpu_stream, ncclComm_t comm,
                    int current_call, int max_call) {
  std::vector<Participant> p;
  {
    tsl::mutex_lock l(Manager::instance().mu);
    if (Manager::instance().collectives.find(comm->id) ==
        Manager::instance().collectives.end()) {
      std::vector<Participant> participants;
      participants.push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      Manager::instance().collectives[comm->id] = participants;
      p = participants;
    } else {
      Manager::instance().collectives[comm->id].push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      p = Manager::instance().collectives[comm->id];
    }

    if (p.size() != comm->nranks) {
      Manager::instance().cv.wait(l);
    } else {
      Manager::instance().collectives.erase(comm->id);
      std::sort(p.begin(), p.end(),
                [](const Participant& a, const Participant& b) -> bool {
                  return a.rank < b.rank;
                });

      se::gpu::GpuStreamHandle stream = p[0].stream;
      if (current_call == 0) stream_wait_streamlist(stream, p);
      if (dtype == PRED)
        allgather_dpcpp<bool>(stream, element_count, p, comm->nranks);
      else if (dtype == F32)
        allgather_dpcpp<float>(stream, element_count, p, comm->nranks);
      else if (dtype == F64)
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
      if (current_call == (max_call - 1)) streamlist_wait_stream(stream, p);
      Manager::instance().cv.notify_all();
    }
  }
}

void sycl_alltoall(std::vector<const void*> send_buffers,
                   std::vector<void*> recv_buffers, int element_count,
                   PrimitiveType dtype, se::gpu::GpuStreamHandle gpu_stream,
                   ncclComm_t comm) {
  std::vector<AlltoAllParticipant> p;
  {
    tsl::mutex_lock l(Manager::instance().mu);
    if (Manager::instance().alltoall_collectives.find(comm->id) ==
        Manager::instance().alltoall_collectives.end()) {
      std::vector<AlltoAllParticipant> participants;
      participants.push_back(
          {gpu_stream, send_buffers, recv_buffers, comm->rank});
      Manager::instance().alltoall_collectives[comm->id] = participants;
      p = participants;
    } else {
      Manager::instance().alltoall_collectives[comm->id].push_back(
          {gpu_stream, send_buffers, recv_buffers, comm->rank});
      p = Manager::instance().alltoall_collectives[comm->id];
    }

    if (p.size() != comm->nranks) {
      Manager::instance().cv.wait(l);
    } else {
      Manager::instance().alltoall_collectives.erase(comm->id);
      std::sort(
          p.begin(), p.end(),
          [](const AlltoAllParticipant& a,
             const AlltoAllParticipant& b) -> bool { return a.rank < b.rank; });

      se::gpu::GpuStreamHandle stream = p[0].stream;
      stream_wait_streamlist(stream, p);
      if (dtype == PRED)
        alltoall_dpcpp<bool>(stream, element_count, p, comm->nranks);
      else if (dtype == F32)
        alltoall_dpcpp<float>(stream, element_count, p, comm->nranks);
      else if (dtype == F64)
        alltoall_dpcpp<double>(stream, element_count, p, comm->nranks);
      else if (dtype == S32)
        alltoall_dpcpp<int32_t>(stream, element_count, p, comm->nranks);
      else if (dtype == S64)
        alltoall_dpcpp<int64_t>(stream, element_count, p, comm->nranks);
      else if (dtype == BF16)
        alltoall_dpcpp<bfloat16>(stream, element_count, p, comm->nranks);
      else if (dtype == U32)
        alltoall_dpcpp<uint32_t>(stream, element_count, p, comm->nranks);
      else if (dtype == U64)
        alltoall_dpcpp<uint64_t>(stream, element_count, p, comm->nranks);
      else
        LOG(FATAL) << "PrimitiveType "
                   << primitive_util::LowercasePrimitiveTypeName(dtype)
                   << " is not supported in AllToAll.";
      streamlist_wait_stream(stream, p);

      Manager::instance().cv.notify_all();
    }
  }
}

void sycl_alltoall_split(std::vector<const void*> send_buffers,
                         std::vector<void*> recv_buffers, int element_count,
                         PrimitiveType dtype,
                         se::gpu::GpuStreamHandle gpu_stream, ncclComm_t comm) {
  std::vector<AlltoAllParticipant> p;
  {
    tsl::mutex_lock l(Manager::instance().mu);
    if (Manager::instance().alltoall_collectives.find(comm->id) ==
        Manager::instance().alltoall_collectives.end()) {
      std::vector<AlltoAllParticipant> participants;
      participants.push_back(
          {gpu_stream, send_buffers, recv_buffers, comm->rank});
      Manager::instance().alltoall_collectives[comm->id] = participants;
      p = participants;
    } else {
      Manager::instance().alltoall_collectives[comm->id].push_back(
          {gpu_stream, send_buffers, recv_buffers, comm->rank});
      p = Manager::instance().alltoall_collectives[comm->id];
    }

    if (p.size() != comm->nranks) {
      Manager::instance().cv.wait(l);
    } else {
      Manager::instance().alltoall_collectives.erase(comm->id);
      std::sort(
          p.begin(), p.end(),
          [](const AlltoAllParticipant& a,
             const AlltoAllParticipant& b) -> bool { return a.rank < b.rank; });

      se::gpu::GpuStreamHandle stream = p[0].stream;
      stream_wait_streamlist(stream, p);
      if (dtype == PRED)
        alltoall_split_dpcpp<bool>(stream, element_count, p, comm->nranks);
      else if (dtype == F32)
        alltoall_split_dpcpp<float>(stream, element_count, p, comm->nranks);
      else if (dtype == F64)
        alltoall_split_dpcpp<double>(stream, element_count, p, comm->nranks);
      else if (dtype == S32)
        alltoall_split_dpcpp<int32_t>(stream, element_count, p, comm->nranks);
      else if (dtype == S64)
        alltoall_split_dpcpp<int64_t>(stream, element_count, p, comm->nranks);
      else if (dtype == U32)
        alltoall_split_dpcpp<uint32_t>(stream, element_count, p, comm->nranks);
      else if (dtype == U64)
        alltoall_split_dpcpp<uint64_t>(stream, element_count, p, comm->nranks);
      else
        LOG(FATAL) << "PrimitiveType "
                   << primitive_util::LowercasePrimitiveTypeName(dtype)
                   << " is not supported in AllToAll.";
      streamlist_wait_stream(stream, p);

      Manager::instance().cv.notify_all();
    }
  }
}

void sycl_reduce_scatter(const void* send_buffer, void* recv_buffer,
                         int element_count, PrimitiveType dtype,
                         ReductionKind reduction_kind,
                         se::gpu::GpuStreamHandle gpu_stream, ncclComm_t comm,
                         int current_call, int max_call) {
  std::vector<Participant> p;
  {
    tsl::mutex_lock l(Manager::instance().mu);
    if (Manager::instance().collectives.find(comm->id) ==
        Manager::instance().collectives.end()) {
      std::vector<Participant> participants;
      participants.push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      Manager::instance().collectives[comm->id] = participants;
      p = participants;
    } else {
      Manager::instance().collectives[comm->id].push_back(
          {gpu_stream, send_buffer, recv_buffer, comm->rank});
      p = Manager::instance().collectives[comm->id];
    }

    if (p.size() != comm->nranks) {
      Manager::instance().cv.wait(l);
    } else {
      Manager::instance().collectives.erase(comm->id);
      std::sort(p.begin(), p.end(),
                [](const Participant& a, const Participant& b) -> bool {
                  return a.rank < b.rank;
                });

      se::gpu::GpuStreamHandle stream = p[0].stream;
      if (current_call == 0) stream_wait_streamlist(stream, p);

      if (reduction_kind == ReductionKind::SUM) {
        if (dtype == PRED)
          reducescatter_dpcpp<bool, sycl::plus<bool>>(stream, element_count, p,
                                                      comm->nranks);
        else if (dtype == F32)
          reducescatter_dpcpp<float, sycl::plus<float>>(stream, element_count,
                                                        p, comm->nranks);
        else if (dtype == F64)
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
        else if (dtype == C64)
          reducescatter_dpcpp<std::complex<float>,
                              sycl::plus<std::complex<float>>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == C128)
          reducescatter_dpcpp<std::complex<double>,
                              sycl::plus<std::complex<double>>>(
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
        else if (dtype == F32)
          reducescatter_dpcpp<float, sycl::multiplies<float>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == F64)
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
        else if (dtype == C64)
          reducescatter_dpcpp<std::complex<float>,
                              sycl::multiplies<std::complex<float>>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == C128)
          reducescatter_dpcpp<std::complex<double>,
                              sycl::multiplies<std::complex<double>>>(
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
        else if (dtype == F32)
          reducescatter_dpcpp<float, sycl::minimum<float>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == F64)
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
        else if (dtype == F32)
          reducescatter_dpcpp<float, sycl::maximum<float>>(
              stream, element_count, p, comm->nranks);
        else if (dtype == F64)
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

      if (current_call == (max_call - 1)) streamlist_wait_stream(stream, p);
      Manager::instance().cv.notify_all();
    }
  }
}

void sycl_collective_permute(const void* send_buffer, void* recv_buffer,
                             int element_count, PrimitiveType dtype,
                             const std::optional<int64_t>& source_id,
                             const std::optional<int64_t>& target_id,
                             se::gpu::GpuStreamHandle gpu_stream,
                             ncclComm_t comm) {
  std::vector<PermuteParticipant> p;
  {
    tsl::mutex_lock l(Manager::instance().mu);
    if (Manager::instance().permute_collectives.find(comm->id) ==
        Manager::instance().permute_collectives.end()) {
      std::vector<PermuteParticipant> participants;
      participants.push_back({gpu_stream, send_buffer, recv_buffer, source_id,
                              target_id, comm->rank});
      Manager::instance().permute_collectives[comm->id] = participants;
      p = participants;
    } else {
      Manager::instance().permute_collectives[comm->id].push_back(
          {gpu_stream, send_buffer, recv_buffer, source_id, target_id,
           comm->rank});
      p = Manager::instance().permute_collectives[comm->id];
    }

    if (p.size() != comm->nranks) {
      Manager::instance().cv.wait(l);
    } else {
      Manager::instance().permute_collectives.erase(comm->id);
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

      Manager::instance().cv.notify_all();
    }
  }
}

}  // namespace gpu
}  // namespace xla
#endif  // ITEX_USE_CCL
