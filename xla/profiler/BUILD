package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sycl_device_tracer",
    srcs = ["device_tracer_sycl.cc"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":ze_tracer",
        "//xla/stream_executor/sycl:hw_info",
        "//xla/stream_executor/sycl:sycl_gpu_runtime",
        "@tsl//tsl/profiler/backends/cpu:annotation_stack",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:trace_events_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/utils:parse_annotation",
        "@tsl//tsl/profiler/utils:trace_utils",
        "@tsl//tsl/profiler/utils:tf_xplane_visitor",
        "@tsl//tsl/profiler/utils:xplane_builder",
        "@tsl//tsl/profiler/utils:xplane_utils",
        "@tsl//tsl/profiler/utils:xplane_schema",
    ],
    alwayslink = True,
)

cc_library(
    name = "ze_tracer",
    hdrs = [
        "trace_options.h",
        "tracing.h",
        "ze_api_collector.h",
        "ze_kernel_collector.h",
        "ze_tracer.h",
        "ze_utils.h",
    ],
    srcs = [
        ":profiler_utils",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":ze_correlator",
        "@tsl//tsl/platform:abi",
    ],
)

cc_library(
    name = "ze_correlator",
    hdrs = [
        "correlator.h",
    ],
    srcs = [
        "correlator.cc",
        ":profiler_utils",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/time",
        "@tsl//tsl/profiler/backends/cpu:annotation_stack",
    ],
)

filegroup(
    name = "profiler_utils",
    srcs = [
        "utils.h",
    ],
    visibility = ["//visibility:public"],
)
