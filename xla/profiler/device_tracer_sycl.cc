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

#include <level_zero/ze_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#include <CL/sycl/backend/level_zero.hpp>
#else
#error "Unsupported compiler"
#endif

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/backends/cpu/annotation_stack.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "tsl/profiler/utils/parse_annotation.h"
#include "tsl/profiler/utils/tf_op_utils.h"
#include "tsl/profiler/utils/xplane_builder.h"
#include "tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/utils/xplane_visitor.h"
#include "xla/profiler/ze_tracer.h"
#include "xla/stream_executor/sycl/hw_info.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

namespace xla {
namespace profiler {

using absl::OkStatus;
using absl::Status;
using tensorflow::ProfileOptions;
using tensorflow::profiler::XEventMetadata;
using tensorflow::profiler::XSpace;
using tsl::profiler::Annotation;
using tsl::profiler::ParseAnnotationStack;
using tsl::profiler::ProfilerInterface;
using tsl::profiler::StatType;
using tsl::profiler::XEventBuilder;
using tsl::profiler::XLineBuilder;
using tsl::profiler::XPlaneBuilder;
using IntelPtiTracer = ZeTracer;

static IntelPtiTracer* g_IntelPtiTracer = nullptr;

inline std::string GpuPlaneName(int32_t device_ordinal) {
  return absl::StrCat("/device:GPU:", device_ordinal);
}

inline std::string ToXStat(const ZeKernelProps& prop) {
  return absl::StrCat(" SIMD width:", prop.simd_width,
                      " grid:", prop.group_count[0], ",", prop.group_count[1],
                      ",", prop.group_count[2], " block:", prop.group_size[0],
                      ",", prop.group_size[1], ",", prop.group_size[2]);
}

static void NormalizeTimeStamps(XPlaneBuilder* plane,
                                uint64_t start_walltime_ns) {
  plane->ForEachLine(
      [&](XLineBuilder line) { line.SetTimestampNs(start_walltime_ns); });
}

class PerDeviceCollector {
 public:
  PerDeviceCollector(int device_id, uint64_t start_walltime_ns,
                     uint64_t start_gpu_ns)
      : start_walltime_ns_(start_walltime_ns), start_gpu_ns_(start_gpu_ns) {
    sycl::device* device_h;
    SYCLGetDevice(&device_h, device_id);
    auto l0_native_queue =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*device_h);
    zePluggableTracerQueueList queue_list = ZeKernelCollector::
        GetzePluggableTracerDeviceQueueMap()[l0_native_queue];
    queues_.assign(queue_list.begin(), queue_list.end());
  }

  void CreateXEvent(const zePluggableTracerEventList& event_list,
                    XPlaneBuilder* plane, XLineBuilder* line) {
    for (const zePluggableTracerEvent& event : event_list) {
      std::string kernel_name = event.kernel_name;
      if (event.append_time + start_gpu_ns_ < start_walltime_ns_) {
        VLOG(2) << "Skip events have abnormal timestamps:" << event.kernel_name
                << " start time(ns): " << event.append_time + start_gpu_ns_
                << " start wall time(ns): " << start_walltime_ns_;
        continue;
      }
      XEventMetadata* event_metadata =
          plane->GetOrCreateEventMetadata(std::move(kernel_name));
      XEventBuilder xevent = line->AddEvent(*event_metadata);
      xevent.SetTimestampNs(event.host_start_time + start_gpu_ns_);
      xevent.SetEndTimestampNs(event.host_end_time + start_gpu_ns_);

      if (event.kernel_props.bytes_transferred > 0) {
        xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                                std::string("Memory transfered bytes")),
                            event.kernel_props.bytes_transferred);
      } else {
        xevent.AddStatValue(
            *plane->GetOrCreateStatMetadata(
                GetStatTypeStr(StatType::kKernelDetails)),
            *plane->GetOrCreateStatMetadata(ToXStat(event.kernel_props)));
      }
      std::vector<Annotation> annotation_stack =
          ParseAnnotationStack(event.annotation);
      if (!annotation_stack.empty()) {
        xevent.AddStatValue(
            *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
            *plane->GetOrCreateStatMetadata(annotation_stack.begin()->name));
      }

      // If multiple metadata have the same key name, show the values from the
      // top of the stack (innermost annotation). Concatenate the values from
      // "hlo_op".
      absl::flat_hash_set<absl::string_view> key_set;

      for (auto annotation = annotation_stack.rbegin();
           annotation != annotation_stack.rend(); ++annotation) {
        for (const Annotation::Metadata& metadata : annotation->metadata) {
          if (key_set.insert(metadata.key).second) {
            xevent.ParseAndAddStatValue(
                *plane->GetOrCreateStatMetadata(metadata.key), metadata.value);
          }
        }
      }
    }
  }

  void CreateHostXEvent(XPlaneBuilder* plane, XLineBuilder* line) const {
    for (zePluggableTracerHostEvent& event :
         ZeTracer::GetzePluggableTracerHostEventList()) {
      if (event.start_time + start_gpu_ns_ < start_walltime_ns_) continue;
      std::string api_name = event.api_name;
      XEventMetadata* event_metadata =
          plane->GetOrCreateEventMetadata(std::move(api_name));
      XEventBuilder xevent = line->AddEvent(*event_metadata);

      xevent.SetTimestampNs(event.start_time + start_gpu_ns_);
      xevent.SetEndTimestampNs(event.end_time + start_gpu_ns_);
    }
  }

  void Flush(XPlaneBuilder* device_plane) {
    tsl::mutex_lock lock(mutex_);
    zePluggableTracerEventMap& event_map =
        ZeKernelCollector::GetzePluggableTracerEventMap();
    for (int i = 0; i < queues_.size(); i++) {
      int64_t line_id = i;
      XLineBuilder line = device_plane->GetOrCreateLine(line_id);
      line.SetTimestampNs(start_walltime_ns_);
      CreateXEvent(event_map[queues_[i]], device_plane, &line);
    }
    {  // Host Runtime API
      int64_t line_id = queues_.size();
      XLineBuilder line = device_plane->GetOrCreateLine(line_id);
      line.SetTimestampNs(start_walltime_ns_);
      CreateHostXEvent(device_plane, &line);
    }

    device_plane->ForEachLine([&](XLineBuilder line) {
      if (line.Id() < queues_.size())
        line.SetName(absl::StrCat("XPU queue/", line.Id()));
      else
        line.SetName("Host Runtime Call");
    });
  }

 private:
  std::vector<ze_command_queue_handle_t> queues_;
  uint64_t start_walltime_ns_;
  uint64_t start_gpu_ns_;
  tsl::mutex mutex_;
};

class PluginTracerSyclImpl : public tsl::profiler::ProfilerInterface {
 public:
  explicit PluginTracerSyclImpl(const ProfileOptions& options)
      : options_(options) {
    LOG(INFO) << "Intel PluginTracerSycl implementation created.";
    if (g_IntelPtiTracer) {
      intelpti_tracer_ = g_IntelPtiTracer;
    }
  }

  ~PluginTracerSyclImpl() override {
    if (intelpti_tracer_) {
      tsl::profiler::AnnotationStack::Enable(false);
      delete intelpti_tracer_;
    }
  }

  // PluginTracerSyclImpl interface:
  Status Start() override {
    LOG(INFO) << "Starting PluginTracerSyclImpl.";
    if (intelpti_tracer_) {
      intelpti_tracer_->Start();
    }
    return OkStatus();
  };
  Status Stop() override {
    LOG(INFO) << "Stopping PluginTracerSyclImpl.";
    return OkStatus();
  }

  Status CollectData(XSpace* space) override {
    LOG(INFO) << "Collecting data to XSpace from PluginTracerSyclImpl.";
    int device_count = 0;
    SYCLGetDeviceCount(&device_count);

    std::vector<xla::profiler::PerDeviceCollector> per_device_collector;
    for (int i = 0; i < device_count; i++) {
      per_device_collector.emplace_back(i, intelpti_tracer_->GetStartWallTime(),
                                        intelpti_tracer_->GetGPUStartTime());
      std::string name = GpuPlaneName(i);
      XPlaneBuilder device_plane(
          tsl::profiler::FindOrAddMutablePlaneWithName(space, name));
      device_plane.SetId(i);
      per_device_collector[i].Flush(&device_plane);
      NormalizeTimeStamps(&device_plane, intelpti_tracer_->GetStartWallTime());
    }

    return OkStatus();
  };

 private:

  IntelPtiTracer* intelpti_tracer_;
  ProfileOptions options_;
};

static bool IsSyclDeviceProfilerEnabled() {
  std::string enable_trace_layer = utils::GetEnv("ZE_ENABLE_TRACING_LAYER");
  std::string use_cycles_per_second = utils::GetEnv("UseCyclesPerSecondTimer");
  if (enable_trace_layer == "1" && use_cycles_per_second == "1") {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<ProfilerInterface> CreatePluginTracer(
    const ProfileOptions& options) {
  if (options.device_tracer_level() == 0) {
    LOG(WARNING) << "Skip CreatePluginTracer since device_tracer_level is 0.";
    return nullptr;
  }

  return std::make_unique<PluginTracerSyclImpl>(options);
}

static auto register_plugin_tracer_factory = [] {
  if (!IsSyclDeviceProfilerEnabled()) {
    LOG(WARNING)
        << "******************************Intel Extension For OpenXLA "
           "profiler "
           "Warning***************************************************";
    LOG(WARNING)
        << "Intel Extension For OpenXLA profiler not enabled, if you want "
           "to enable it, please set "
           "environment as :\nexport ZE_ENABLE_TRACING_LAYER=1 \nexport "
           "UseCyclesPerSecondTimer=1\n";
    LOG(WARNING) << "*****************************************************"
                    "*****************"
                    "********************************";
    return 0;
  }

  assert(zeInit(ZE_INIT_FLAG_GPU_ONLY) == ZE_RESULT_SUCCESS);

  std::string enable_immediate_commmand_list =
      utils::GetEnv("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS");
  if (enable_immediate_commmand_list == "0") {
    utils::ImmediateCommandListDisabled();
  } else if (enable_immediate_commmand_list.empty()) {
    if (!IsXeHPC()) {
      utils::ImmediateCommandListDisabled();
    }
  }

  uint32_t flags = 0;
  flags |= (1 << TRACE_DEVICE_TIMING);
  flags |= (1 << TRACE_HOST_RUNTIME_TIMING);
  const char* env_value = std::getenv("DisableHostRuntimeTimer");
  if (env_value) {
    auto disable_host_runtime_timing = absl::AsciiStrToLower(env_value);
    if (disable_host_runtime_timing == "1" ||
        disable_host_runtime_timing == "true") {
      LOG(INFO) << "Intel Extension For OpenXLA profiler enabled without "
                   "tracing host runtime API.";
      flags ^= (1 << TRACE_HOST_RUNTIME_TIMING);
    }
  }

  g_IntelPtiTracer = ZeTracer::Create(TraceOptions(flags));
  tsl::profiler::AnnotationStack::Enable(true);
  RegisterProfilerFactory(&CreatePluginTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace xla
