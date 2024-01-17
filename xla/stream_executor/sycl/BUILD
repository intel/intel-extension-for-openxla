load("//xla:xla.bzl", "xpu_library")
load(
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

xpu_library(
    name = "sycl_gpu_runtime",
    srcs = ["sycl_gpu_runtime.cc"],
    hdrs = ["sycl_gpu_runtime.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@tsl//tsl/util:env_var",
        "@local_config_sycl//sycl:sycl_headers",
    ],
)

cc_library(
    name = "hw_info",
    srcs = ["hw_info.cc"],
    hdrs = [
        "hw_info.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":sycl_gpu_runtime",
    ],
)

cc_library(
    name = "sycl_driver",
    srcs = ["sycl_driver.cc"],
    deps = [
        ":sycl_gpu_runtime",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@xla//xla/stream_executor:stream_executor_headers",
        "@xla//xla/stream_executor:device_options",
        "@xla//xla/stream_executor/gpu:gpu_driver_header",
        "@xla//xla/stream_executor/platform",
        "@xla//xla/stream_executor/platform:dso_loader",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:stacktrace",
        "@tsl//tsl/platform:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ] + if_sycl_is_configured([
        "@local_config_sycl//sycl:mkl",
    ]),
)

cc_library(
    name = "sycl_platform_id",
    srcs = ["sycl_platform_id.cc"],
    hdrs = ["sycl_platform_id.h"],
    deps = ["@xla//xla/stream_executor:platform"],
)

cc_library(
    name = "sycl_platform",
    srcs = ["sycl_platform.cc"],
    hdrs = ["sycl_platform.h"],
    visibility = ["//visibility:public"],
    deps =
        [
            ":sycl_executor",
            ":sycl_platform_id",
            "@xla//xla/stream_executor",  # buildcleaner: keep
            "@xla//xla/stream_executor:executor_cache",
            "@xla//xla/stream_executor:multi_platform_manager",
            "@xla//xla/stream_executor:stream_executor_headers",
            "@xla//xla/stream_executor/platform",
            "@xla//xla/stream_executor/gpu:gpu_executor_header",
        ] + [
            "@com_google_absl//absl/base",
            "@com_google_absl//absl/base:core_headers",
            "@com_google_absl//absl/memory",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
            "@com_google_absl//absl/synchronization",
        ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "sycl_kernel",
    srcs = ["sycl_kernel.cc"],
    hdrs = ["sycl_kernel.h"],
    deps = [
        ":sycl_gpu_runtime",
        "@xla//xla/stream_executor:stream_executor_headers",
        "@xla//xla/stream_executor/gpu:gpu_kernel_header",
        "@xla//xla/stream_executor/platform",
    ],
)

cc_library(
    name = "sycl_event",
    srcs = ["sycl_event.cc"],
    hdrs = ["sycl_event.h"],
    deps = [
        ":sycl_driver",
        ":sycl_stream",
        "@xla//xla/stream_executor:stream_executor_headers",
        "@xla//xla/stream_executor/gpu:gpu_event",
        "@xla//xla/stream_executor/gpu:gpu_stream_header",
    ],
)

cc_library(
    name = "sycl_stream",
    hdrs = ["sycl_stream.h"],
    deps = [
        ":sycl_driver",
        "@xla//xla/stream_executor:stream_executor_headers",
        "@xla//xla/stream_executor/gpu:gpu_stream",
        "@xla//xla/stream_executor/platform",
    ],
)

cc_library(
    name = "sycl_executor",
    srcs = ["sycl_executor.cc"],
    hdrs = ["sycl_executor.h"],
    deps = [
        ":sycl_driver",
        ":sycl_event",
        ":sycl_kernel",
        ":sycl_platform_id",
        ":sycl_stream",
        "@com_google_absl//absl/strings",
        "@xla//xla/stream_executor:event",
        "@xla//xla/stream_executor:stream_executor_internal",
        "@xla//xla/stream_executor:stream_executor_headers",
        "@xla//xla/stream_executor/platform",
        "@xla//xla/stream_executor/platform:dso_loader",
    ] + [
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = True,
)

cc_library(
    name = "all_runtime",
    visibility = ["//visibility:public"],
    deps = [
        ":sycl_platform",
    ],
    alwayslink = 1,
)
