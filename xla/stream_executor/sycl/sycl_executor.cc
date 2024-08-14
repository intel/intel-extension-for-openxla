/* Copyright (c) 2023 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/stream_executor/sycl/sycl_executor.h"

#include <unistd.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/statusor.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/fingerprint.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/hw_info.h"
#include "xla/stream_executor/sycl/sycl_event.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/stream_executor/sycl/sycl_stream.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"

namespace stream_executor {
namespace gpu {

GpuContext* ExtractGpuContext(GpuExecutor* gpu_exec) {}

namespace sycl = ::sycl;

// Hook that can be used to CUBIN-ate PTX before it is loaded into the driver.
// It has been observed that loading both PTX and cubins into the driver library
// can cause it to crash, but loading only CUBINs avoids those crashes;
// therefore, it's useful to have this hook to hack in uniform CUBIN-ation of
// PTX code.
//
// As this is an implementation-detail workaround, the usage is to declare this
// variable with extern linkage and populate it from another translation unit.
std::function<std::string(const std::string&)> g_cubinate;

static GpuEvent* AsGpuEvent(Event* event) {
  DCHECK(event != nullptr);
  return static_cast<GpuEvent*>(event);
}

static void* AsSyclDevicePtr(const DeviceMemoryBase& gpu_mem) {
  return const_cast<void*>(gpu_mem.opaque());
}

// See description on const version above.
static void* AsSyclDevicePtr(DeviceMemoryBase* gpu_mem) {
  return AsSyclDevicePtr(*gpu_mem);
}

GpuExecutor::~GpuExecutor() {
  CHECK(kernel_to_gpu_binary_.empty()) << "GpuExecutor has live kernels.";
  CHECK(gpu_binary_to_module_.empty()) << "GpuExecutor has loaded modules.";
  if (context_ != nullptr) {
    GpuDriver::DestroyContext(context_);
  }
}

absl::Status GpuExecutor::Init() {
  auto status = GpuDriver::GetDevice(device_ordinal_, &device_);
  if (!status.ok()) {
    return status;
  }

  return GpuDriver::CreateContext(device_ordinal_, device_, &context_);
}

absl::Status GpuExecutor::LoadModuleFromCuBin(const char* cubin,
                                             ze_module_handle_t* module) {
  LOG(FATAL) << "Feature not supported on SYCL platform (LoadModuleFromCuBin)";
}

absl::Status GpuExecutor::LoadModuleFromPtx(const char* ptx,
                                           ze_module_handle_t* module) {
  LOG(FATAL) << "Feature not supported on SYCL platform (LoadModuleFromPtx)";
}

absl::Status GpuExecutor::LoadModuleFromHsaco(const char* hsaco,
                                             ze_module_handle_t* module) {
  LOG(FATAL) << "Feature not supported on SYCL platform (LoadModuleFromHsaco)";
}

absl::Status GpuExecutor::LoadModuleFromSpir(const char* spirv,
                                            const size_t size,
                                            ze_module_handle_t* module) {
  uint64_t module_refcount;
  std::tie(*module, module_refcount) = gpu_binary_to_module_[spirv];

  if (*module == nullptr) {
    TF_RETURN_IF_ERROR(GpuDriver::LoadLevelzero(context_, spirv, size, module));

    module_refcount = 1;
    VLOG(3) << "Loaded SPIRV " << static_cast<const void*>(spirv)
            << " as module " << *module;
  } else {
    ++module_refcount;
    VLOG(3) << "SPIRV " << static_cast<const void*>(spirv)
            << " is already loaded as module " << *module;
  }
  gpu_binary_to_module_[spirv] = {*module, module_refcount};
  return absl::OkStatus();
}

absl::Status GpuExecutor::GetKernel(const MultiKernelLoaderSpec& spec,
                                    Kernel* kernel) {
  GpuKernel* l0_kernel = AsGpuKernel(kernel);
  ze_module_handle_t module = nullptr;
  string kernel_name;

  if (spec.has_cuda_cubin_in_memory()) {
    kernel_name = spec.cuda_cubin_in_memory().kernel_name();
    const char* spirv = reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().cubin_bytes().data());
    int size = spec.cuda_cubin_in_memory().size();
    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_RETURN_IF_ERROR(LoadModuleFromSpir(spirv, size, &module));
    kernel_to_gpu_binary_[kernel] = spirv;
  } else {
    return absl::Status(
        absl::StatusCode::kInternal,
        absl::StrFormat("No method of loading SPIR kernel provided"));
  }

  VLOG(2) << "getting function " << kernel_name << " from module " << module;
  TF_RETURN_IF_ERROR(GpuDriver::GetModuleFunction(
      context_, module, kernel_name.c_str(), l0_kernel->gpu_function_ptr()));

  // We have to trust the kernel loader spec arity because there doesn't
  // appear to be a way to reflect on the number of expected arguments w/the
  // SPIR API.
  l0_kernel->set_arity(spec.arity());

  KernelMetadata kernel_metadata;
  TF_RETURN_IF_ERROR(GetKernelMetadata(l0_kernel, &kernel_metadata));
  kernel->set_metadata(kernel_metadata);
  kernel->set_name(kernel_name);
  return absl::OkStatus();
}

bool GpuExecutor::UnloadGpuBinary(const void* gpu_binary) {
  auto module_it = gpu_binary_to_module_.find(gpu_binary);
  if (gpu_binary_to_module_.end() == module_it) {
    VLOG(3) << "No loaded  SPIR module for " << gpu_binary;
    return false;
  }
  auto& module = module_it->second.first;
  auto& refcount = module_it->second.second;
  VLOG(3) << "Found SPIR module " << module << " with refcount " << refcount;
  if (--refcount == 0) {
    VLOG(3) << "Unloading  SPIR module " << module;
    GpuDriver::UnloadModule(context_, module);
    gpu_binary_to_module_.erase(module_it);
  }
  return true;
}

void GpuExecutor::UnloadKernel(const Kernel* kernel) {
  VLOG(3) << "Unloading kernel " << kernel << " : " << kernel->name();

  absl::MutexLock lock{&in_memory_modules_mu_};
  auto gpu_binary_it = kernel_to_gpu_binary_.find(kernel);
  if (kernel_to_gpu_binary_.end() == gpu_binary_it) {
    VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
            << " has never been loaded.";
    return;  // We've never seen this kernel.
  }
  VLOG(3) << "Kernel " << kernel << " : " << kernel->name()
          << " has loaded GPU code " << gpu_binary_it->second;
  UnloadGpuBinary(gpu_binary_it->second);
  kernel_to_gpu_binary_.erase(gpu_binary_it);
}

absl::Status GpuExecutor::LoadModule(const MultiModuleLoaderSpec& spec,
                                    ModuleHandle* module_handle) {
  ze_module_handle_t ze_module = nullptr;
  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};

    TF_RETURN_IF_ERROR(LoadModuleFromSpir(
        reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data()),
        spec.cuda_cubin_in_memory().size(), &ze_module));
    *module_handle = ModuleHandle(const_cast<void*>(
        static_cast<const void*>(spec.cuda_cubin_in_memory().data())));
    return absl::OkStatus();
  } else {
    return absl::Status(absl::StatusCode::kInternal,
                       absl::StrFormat("No SPIR binary found"));
  }
}

bool GpuExecutor::UnloadModule(ModuleHandle module_handle) {
  const char* gpu_binary = reinterpret_cast<const char*>(module_handle.id());
  absl::MutexLock lock{&in_memory_modules_mu_};
  return UnloadGpuBinary(gpu_binary);
}

namespace {
absl::uint128 Fingerprint128(const absl::string_view s) {
  auto fp = tsl::Fingerprint128(s);
  return absl::MakeUint128(fp.high64, fp.low64);
}
}  // namespace

absl::StatusOr<std::shared_ptr<DeviceMemoryBase>>
GpuExecutor::CreateOrShareConstant(Stream* stream,
                                   absl::Span<const uint8_t> content) {
  absl::MutexLock lock{&shared_constants_mu_};
  // We assume all constants are uniquely identified by this hash. In the
  // (highly unlikely) event of a hash collision, the program will likely crash
  // (because the cached constant that will be returned by mistake is unlikely
  // to have the correct size).
  absl::uint128 fingerprint = Fingerprint128(absl::string_view(
      reinterpret_cast<const char*>(content.data()), content.size()));
  // Must insert nullptr first to get an iterator to the insertion point.
  auto insert_result = shared_constants_.insert(
      {fingerprint, std::weak_ptr<DeviceMemoryBase>()});
  auto it = insert_result.first;
  bool was_already_in_cache = !insert_result.second;
  std::shared_ptr<DeviceMemoryBase> shared_constant;

  if (was_already_in_cache) {
    shared_constant = it->second.lock();
  }

  if (shared_constant == nullptr) {
    // Either the constant wasn't found in the cache, or it was but its
    // weak_ptr had expired.
    DeviceMemoryBase* new_constant =
        new DeviceMemoryBase(Allocate(content.size(), /*memory_space=*/0));
    if (new_constant->opaque() == nullptr) {
      return absl::Status(
          absl::StatusCode::kInternal,
          absl::StrFormat("Failed to allocate %d bytes for new constant",
                          content.size()));
    }

    TF_RETURN_IF_ERROR(
        stream->Memcpy(new_constant, content.data(), content.size()));
    absl::Status status = stream->BlockHostUntilDone();

    if (!status.ok()) {
      Deallocate(new_constant);
      return absl::Status(absl::StatusCode::kInternal,
                          absl::StrFormat("Memcpy to device address %p failed",
                                         new_constant->opaque()));
    }

    // Capturing 'this' in the custom deleter means this executor must
    // outlive all shared uses of this constant.
    shared_constant = std::shared_ptr<DeviceMemoryBase>(
        new_constant, [this](DeviceMemoryBase* p) {
          Deallocate(p);
          delete p;
        });
    it->second = std::weak_ptr<DeviceMemoryBase>(shared_constant);
  }

  return shared_constant;
}

absl::Status GpuExecutor::GetKernelMetadata(GpuKernel* l0_kernel,
                                           KernelMetadata* kernel_metadata) {
  int value = 0;
  // TODO: implement this feature in SPIR
  kernel_metadata->set_registers_per_thread(value);
  kernel_metadata->set_shared_memory_bytes(value);
  return absl::OkStatus();
}

absl::Status GpuExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const ClusterDim& cluster_dims,
                                 const Kernel& kernel, const KernelArgs& args) {
  if (cluster_dims.x != 1 || cluster_dims.y != 1 || cluster_dims.z != 1)
    return absl::UnimplementedError("Launch is not implemented for SYCL");
  return Launch(stream, thread_dims, block_dims, kernel, args);
}

absl::Status GpuExecutor::Launch(Stream* stream, const ThreadDim& thread_dims,
                                 const BlockDim& block_dims,
                                 const Kernel& kernel, const KernelArgs& args) {
  CHECK_EQ(kernel.Arity() + (args.number_of_shared_bytes() > 0),
           args.number_of_arguments());
  GpuStreamHandle gpu_stream = AsGpuStreamValue(stream);
  const GpuKernel* spir_kernel = AsGpuKernel(&kernel);
  GpuFunctionHandle sycl_kernel = spir_kernel->AsGpuFunctionHandle();

  // Only perform/print the occupancy check once.  Even just checking to see
  // whether we've done an occupancy check on this kernel before isn't free
  // (because we have to synchronize), so we only do this at -v 2+.
  if (VLOG_IS_ON(2)) {
    absl::MutexLock lock(&launched_kernels_mu_);
    if (launched_kernels_.count(sycl_kernel) == 0) {
      launched_kernels_.insert(sycl_kernel);
    }
  }

  auto* packed_args = DynCast<KernelArgsPackedArrayBase>(&args);
  if (!packed_args)
    return absl::InternalError("Unsupported kernel arguments type");

  std::vector<void*> kernargs;
  for (const void *const arg: packed_args->argument_addresses()) {
    kernargs.push_back(reinterpret_cast<void*>(*static_cast<const uint64_t*>(arg)));
  }
  size_t size = kernargs.size();
  void* config[] = {kernargs.data(), &size};

  return GpuDriver::LaunchKernel(
      context_, kernel.name(), sycl_kernel, block_dims.x, block_dims.y,
      block_dims.z, thread_dims.x, thread_dims.y, thread_dims.z,
      args.number_of_shared_bytes(), gpu_stream, nullptr, (void**)&config);
}

absl::Status GpuExecutor::Submit(Stream* stream,
                                const CommandBuffer& command_buffer) {
  LOG(FATAL) << "Submit is not implemented in sycl_executor";
}

DeviceMemoryBase GpuExecutor::Allocate(uint64_t size, int64_t memory_space) {
  CHECK_EQ(memory_space, 0);
  return DeviceMemoryBase(GpuDriver::DeviceAllocate(context_, size), size);
}

void GpuExecutor::Deallocate(DeviceMemoryBase* mem) {
  GpuDriver::DeviceDeallocate(context_, mem->opaque());
}

bool GpuExecutor::SynchronizeAllActivity() {
  return GpuDriver::SynchronizeContext(context_);
}

absl::Status GpuExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                            uint64_t size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return GpuDriver::SynchronousMemsetUint32(
        context_, AsSyclDevicePtr(location), 0x0, size / 4);
  }
  return GpuDriver::SynchronousMemsetUint8(context_, AsSyclDevicePtr(location),
                                           0x0, size);
}

absl::Status GpuExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                           const void* host_src,
                                           uint64_t size) {
  return GpuDriver::SynchronousMemcpyH2D(context_, AsSyclDevicePtr(gpu_dst),
                                         host_src, size);
}

absl::Status GpuExecutor::SynchronousMemcpy(void* host_dst,
                                           const DeviceMemoryBase& gpu_src,
                                           uint64_t size) {
  return GpuDriver::SynchronousMemcpyD2H(context_, host_dst,
                                         AsSyclDevicePtr(gpu_src), size);
}

absl::Status GpuExecutor::MemZero(Stream* stream, DeviceMemoryBase* location,
                                 uint64_t size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
      size % 4 == 0) {
    return Memset32(stream, location, 0x0, size);
  } else {
    return Memset(stream, location, 0x0, size);
  }
}

absl::Status GpuExecutor::Memset(Stream* stream, DeviceMemoryBase* location,
                                uint8_t pattern, uint64_t size) {
  VLOG(2) << "enqueueing memset8 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  return GpuDriver::AsynchronousMemsetUint8(context_, AsSyclDevicePtr(location),
                                            pattern, size,
                                            AsGpuStreamValue(stream));
}

absl::Status GpuExecutor::Memset32(Stream* stream, DeviceMemoryBase* location,
                                  uint32_t pattern, uint64_t size) {
  VLOG(2) << "enqueueing memset32 operation onto stream " << stream
          << " at location " << location << " with size " << size
          << " and pattern " << std::hex << pattern;
  CHECK(reinterpret_cast<uintptr_t>(location->opaque()) % 4 == 0 &&
        size % 4 == 0);
  return GpuDriver::AsynchronousMemsetUint32(
      context_, AsSyclDevicePtr(location), pattern, size / 4,
      AsGpuStreamValue(stream));
}

absl::Status GpuExecutor::Memcpy(Stream* stream, void* host_dst,
                                 const DeviceMemoryBase& gpu_src, uint64_t size) {
  bool ok = GpuDriver::AsynchronousMemcpyD2H(context_, host_dst,
                                             AsSyclDevicePtr(gpu_src), size,
                                             AsGpuStreamValue(stream));
    // TODO(b/326130105): Change AsynchronousMemcpyD2H calls to return Status.
  if (!ok) {
    return absl::InternalError("Failed to memcpy from device to host.");
  }
  return absl::OkStatus();
}

absl::Status GpuExecutor::Memcpy(Stream* stream, DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64_t size) {
  bool ok = GpuDriver::AsynchronousMemcpyH2D(context_, AsSyclDevicePtr(gpu_dst),
                                             host_src, size,
                                             AsGpuStreamValue(stream));
  // TODO(b/326130105): Change AsynchronousMemcpyD2H calls to return Status.
  if (!ok) {
    return absl::InternalError("Failed to memcpy from device to host.");
  }
  return absl::OkStatus();
}

bool GpuExecutor::MemcpyDeviceToDevice(Stream* stream,
                                       DeviceMemoryBase* gpu_dst,
                                       const DeviceMemoryBase& gpu_src,
                                       uint64_t size) {
  return GpuDriver::AsynchronousMemcpyD2D(context_, AsSyclDevicePtr(gpu_dst),
                                          AsSyclDevicePtr(gpu_src), size,
                                          AsGpuStreamValue(stream));
}

bool GpuExecutor::HostCallback(Stream* stream,
                               absl::AnyInvocable<absl::Status() &&> callback) {
  auto callback_ptr_new =
      new absl::AnyInvocable<void() &&>([cb = std::move(callback)]() mutable {
        absl::Status s = std::move(cb)();
        if (!s.ok()) {
          LOG(WARNING) << "Host callback failed: " << s;
        }
      });
  auto ptr_new = reinterpret_cast<void*>(callback_ptr_new);
  auto callback_function = std::function<void()>([ptr_new]() {
    auto* callback_ptr =
        reinterpret_cast<absl::AnyInvocable<void() &&>*>(ptr_new);
    std::move (*callback_ptr)();
    delete callback_ptr;
  });

  sycl::queue* stream_handle = AsGpuStreamValue(stream);
  stream_handle->submit(
      [&](auto& cgh) { cgh.host_task(std::move(callback_function)); });
  return true;
}

void GpuExecutor::DeallocateStream(Stream* stream) {
  GpuStream* gpu_stream = AsGpuStream(stream);
  absl::MutexLock l(&alive_gpu_streams_mu_);
  alive_gpu_streams_.erase(gpu_stream->platform_specific_stream());
  gpu_stream->Destroy();
}

absl::Status GpuExecutor::BlockHostUntilDone(Stream* stream) {
  sycl::queue* stream_handle = AsGpuStreamValue(stream);
  stream_handle->wait();
  return absl::OkStatus();
}

blas::BlasSupport* GpuExecutor::AsBlas() {
  absl::MutexLock lock(&mu_);
  if (blas_ != nullptr) {
    return blas_.get();
  }

  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(stream_executor::sycl::kSyclPlatformId);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().message();
    return nullptr;
  }

  auto blas = status.value()(this);
  blas_.reset(blas);
  return blas_.get();
}

dnn::DnnSupport* GpuExecutor::AsDnn() {
  absl::MutexLock lock(&mu_);
  if (dnn_ != nullptr) {
    return dnn_.get();
  }

  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::DnnFactory> status =
      registry->GetFactory<PluginRegistry::DnnFactory>(stream_executor::sycl::kSyclPlatformId);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve DNN factory: "
               << status.status().message();
    return nullptr;
  }

  auto dnn = status.value()(this);
  dnn_.reset(dnn);
  return dnn_.get();
}

fft::FftSupport* GpuExecutor::AsFft() {
  absl::MutexLock lock(&mu_);
  if (fft_ != nullptr) {
    return fft_.get();
  }

  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(stream_executor::sycl::kSyclPlatformId);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().message();
    return nullptr;
  }

  auto fft = status.value()(this);
  fft_.reset(fft);
  return fft_.get();
}

bool GpuExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
  return false;
}

absl::Status GpuExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  LOG(FATAL) << "Peer access is not supported on SYCL platform";
}

bool GpuExecutor::DeviceMemoryUsage(int64_t* free, int64_t* total) const {
  return GpuDriver::GetDeviceMemoryInfo(context_, free, total);
}

absl::StatusOr<DeviceMemoryBase> GpuExecutor::GetSymbol(
  const std::string& symbol_name, ModuleHandle module_handle) {

  void* mem = nullptr;
  size_t bytes = 0;
  CHECK(static_cast<bool>(module_handle));

  {  // give limited scope to mutex_lock
    absl::MutexLock lock{&in_memory_modules_mu_};
    auto it = gpu_binary_to_module_.find(module_handle.id());
    CHECK(it != gpu_binary_to_module_.end());

    GpuModuleHandle gpu_module_handle = it->second.first;
    CHECK(gpu_module_handle != nullptr);
    TF_RETURN_IF_ERROR(GpuDriver::GetModuleSymbol(
        context_, gpu_module_handle, symbol_name.c_str(),
        &mem, &bytes));
    return DeviceMemoryBase(mem, bytes);
  }

  return absl::NotFoundError(
      absl::StrCat("Check if module containing symbol ", symbol_name,
                   " is loaded (module_handle = ",
                   reinterpret_cast<uintptr_t>(module_handle.id()), ")"));
}
absl::Status FillBlockDimLimit(GpuDeviceHandle device,
                              BlockDim* block_dim_limit) {
  // The BlockDim name is a mismatch against these GRID_DIM_* queries because
  // we use BlockDims to express the dimensions of blocks within a grid
  // (as opposed to ThreadDim which expresses the dimensions of threads
  // within a block).
  int x, y, z;
  TF_RETURN_IF_ERROR(GpuDriver::GetGridLimits(&x, &y, &z, device));
  block_dim_limit->x = x;
  block_dim_limit->y = y;
  block_dim_limit->z = z;
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Kernel>> GpuExecutor::CreateKernel() {
  return std::make_unique<GpuKernel>(this);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>> GpuExecutor::CreateCommandBuffer(
    CommandBuffer::Mode mode) {
  return absl::UnimplementedError("CreateCommandBuffer is not implemented for SYCL");
}

std::unique_ptr<GpuCommandBuffer> GpuExecutor::CreateCommandBuffer(
    CommandBuffer::Mode mode, GpuGraphHandle graph, bool is_owned_graph) {
  VLOG(2) << "Create HIP command buffer (HIP graph) from existing graph "
          << graph << "; is_owned_graph=" << is_owned_graph;
  return std::make_unique<GpuCommandBuffer>(mode, /*parent=*/this, graph,
                                            is_owned_graph);
}

GpuContext* GpuExecutor::gpu_context() { return context_; }

int fpus_per_core(const GpuDeviceHandle& device) {
  int eu_count_per_slice =
      device->template get_info<sycl::ext::intel::info::device::gpu_eu_count>();
  int core_count = GpuDriver::GetMultiprocessorCount(device).value();
  int eu_count_per_core = eu_count_per_slice / core_count;

  // operations/EU/clk
  int n = 0;
  if (IsXeHPC(device)) {
    n = 16;
  } else if (IsXeHPG(device)) {
    n = 8;
  }
  assert(n != 0);

  return eu_count_per_core * n;
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
GpuExecutor::CreateDeviceDescription(int device_ordinal) {
  GpuDeviceHandle device;
  TF_RETURN_IF_ERROR(GpuDriver::GetDevice(device_ordinal, &device));

  internal::DeviceDescriptionBuilder builder;

  int32_t max_workgroup_size =
      device->template get_info<sycl::info::device::max_work_group_size>();
  builder.set_threads_per_block_limit(max_workgroup_size);

  float clock_ghz =
      device->template get_info<sycl::info::device::max_clock_frequency>() /
      1000.;
  builder.set_clock_rate_ghz(clock_ghz);

  uint64_t device_memory_size = static_cast<uint64_t>(-1);
  (void)GpuDriver::GetDeviceTotalMemory(device, &device_memory_size);
  builder.set_device_memory_size(device_memory_size);

  int global_mem_cache_size =
      device->template get_info<sycl::info::device::global_mem_cache_size>();
  builder.set_l2_cache_size(global_mem_cache_size);

  int32_t memory_clock_khz = device->template get_info<
      sycl::ext::intel::info::device::memory_clock_rate>();
  int32_t memory_bus_width = device->template get_info<
      sycl::ext::intel::info::device::memory_bus_width>();
  builder.set_memory_bandwidth(2 * memory_clock_khz * 1e6 * memory_bus_width /
                               8);

  {
    BlockDim block_dim_limit;
    TF_RETURN_IF_ERROR(FillBlockDimLimit(device, &block_dim_limit));
    builder.set_block_dim_limit(block_dim_limit);
  }

  {
    std::string device_name;
    TF_RETURN_IF_ERROR(GpuDriver::GetDeviceName(device, &device_name));
    builder.set_name(device_name);
  }

  builder.set_device_vendor("INTEL Corporation");
  // This means AMPERE.
  builder.set_cuda_compute_capability(8, 0);
  builder.set_shared_memory_per_core(
      GpuDriver::GetMaxSharedMemoryPerCore(device).value());
  builder.set_shared_memory_per_block(
      GpuDriver::GetMaxSharedMemoryPerBlock(device).value());
  int core_count = GpuDriver::GetMultiprocessorCount(device).value();
  builder.set_core_count(core_count);
  builder.set_fpus_per_core(fpus_per_core(device));
  builder.set_threads_per_core_limit(max_workgroup_size);
  builder.set_threads_per_warp(32);

  return builder.Build();
}

}  // namespace gpu
}  // namespace stream_executor
