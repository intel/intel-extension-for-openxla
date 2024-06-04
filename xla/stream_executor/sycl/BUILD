load("//xla:xla.bzl", "xpu_library")
load(
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
)
load("@tsl//tsl/platform:build_config_root.bzl", "if_static")

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
        "@local_config_sycl//sycl:level_zero",
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
    name = "onednn_plugin",
    srcs = if_sycl_is_configured(["sycl_dnn.cc"]),
    hdrs = if_sycl_is_configured(["sycl_dnn.h"]),
    visibility = ["//visibility:public"],
    deps = if_sycl_is_configured([
        ":sycl_driver",
        ":sycl_executor",
        ":sycl_platform_id",
        "@xla//xla/stream_executor",
        "@xla//xla/stream_executor:dnn",
        "@xla//xla/stream_executor:plugin_registry",
        "@xla//xla/stream_executor:stream_executor_internal",
        "@xla//xla/stream_executor/gpu:gpu_executor_header",
        "@xla//xla/stream_executor/gpu:gpu_timer_header",
        "@xla//xla/stream_executor/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ]),
    alwayslink = True,
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
    name = "sycl_collectives",
    srcs = if_sycl_is_configured(["sycl_collectives.cc"]),
    deps = if_sycl_is_configured([
        "@xla//xla/stream_executor/gpu:gpu_collectives_header",
        "@xla//xla/stream_executor/gpu:gpu_driver_header",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
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
    name = "sycl_conditional_kernels",
    srcs = ["sycl_conditional_kernels.cc"],
    deps = [
        "@com_google_absl//absl/log",
    ],
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
        ":hw_info",
        ":sycl_driver",
        ":sycl_event",
        ":sycl_kernel",
        ":sycl_platform_id",
        ":sycl_stream",
        ":sycl_collectives",
        "@com_google_absl//absl/strings",
        "@xla//xla/stream_executor:event",
        "@xla//xla/stream_executor:plugin_registry",
        "@xla//xla/stream_executor:stream_executor_internal",
        "@xla//xla/stream_executor:stream_executor_headers",
        "@xla//xla/stream_executor/platform",
        "@xla//xla/stream_executor/platform:dso_loader",
        "@xla//xla/stream_executor/gpu:gpu_command_buffer",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:fingerprint",
    ],
    alwayslink = True,
)

cc_library(
    name = "syclblas_plugin",
    srcs = ["sycl_blas.cc"],
    hdrs = ["sycl_blas.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sycl_executor",
        ":sycl_platform_id",
        "@xla//xla/stream_executor:blas",
        "@xla//xla/stream_executor:plugin_registry",
        "@xla//xla/stream_executor/gpu:gpu_stream",
        "@xla//xla/stream_executor/gpu:gpu_executor_header",
        "@xla//xla/stream_executor/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_config_sycl//sycl:sycl_headers",
    ] + if_sycl_is_configured([
        "@local_config_sycl//sycl:mkl",
    ]),
    alwayslink = True,
)

cc_library(
    name = "syclfft_plugin",
    srcs = ["sycl_fft.cc"],
    hdrs = ["sycl_fft.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sycl_executor",
        ":sycl_platform_id",
        "@xla//xla/stream_executor:fft",
        "@xla//xla/stream_executor:plugin_registry",
        "@xla//xla/stream_executor/gpu:gpu_stream",
        "@xla//xla/stream_executor/gpu:gpu_executor_header",
        "@xla//xla/stream_executor/gpu:gpu_helpers_header",
        "@xla//xla/stream_executor/platform",
        "@local_config_sycl//sycl:sycl_headers",
    ] + if_sycl_is_configured([
        "@local_config_sycl//sycl:mkl",
    ]),
    alwayslink = True,
)

cc_library(
    name = "all_runtime",
    visibility = ["//visibility:public"],
    deps = [
        ":onednn_plugin",
        ":sycl_platform",
        ":syclblas_plugin",
        ":syclfft_plugin",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sycl_rpath",
    data = [],
    linkopts = select({
        "//conditions:default": [
            "-Wl,-rpath,../local_config_sycl/sycl/sycl/lib",
        ],
    }),
    deps = [],
)

cc_library(
    name = "stream_executor_sycl",
    deps = [
        ":sycl_rpath",
        "@xla//xla/stream_executor:stream_executor_bundle",
    ] + if_static(
        [":all_runtime"],
    ),
)
