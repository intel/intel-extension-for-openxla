load("//xla:xla.bzl", "itex_xpu_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "itex_gpu_header",
    srcs = ["itex_gpu_runtime.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

itex_xpu_library(
    name = "itex_gpu_runtime_imp",
    srcs = ["itex_gpu_runtime.cc"],
    deps = [
        ":itex_gpu_header",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/util:env_var",
    ],
    alwayslink = True,
)

cc_library(
    name = "bfc_allocator",
    srcs = ["bfc_allocator.cc"],
    hdrs = [
        "allocator.h",
        "bfc_allocator.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":itex_gpu_header",
        "@com_google_absl//absl/container:flat_hash_set",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:mutex",
        "@tsl//tsl/platform/default:types",
        "@tsl//tsl/util:env_var",
    ],
    alwayslink = True,
)

cc_library(
    name = "gpu_pool_allocator",
    hdrs = [
        "gpu_pool_allocator.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":bfc_allocator",
        ":itex_gpu_header",
    ],
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
            "@xla//xla/stream_executor:stream_executor_pimpl_header",
            "@xla//xla/stream_executor/platform",
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
    name = "sycl_types_header",
    hdrs = ["sycl_types.h"],
    deps = [
        "@xla//xla/stream_executor/platform",
    ],
)

cc_library(
    name = "sycl_executor_header",
    textual_hdrs = ["sycl_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sycl_kernel",
        "@tsl//tsl/platform:fingerprint",
        "@xla//xla/stream_executor:event",
        "@xla//xla/stream_executor/platform",
    ],
)

cc_library(
    name = "sycl_kernel",
    srcs = ["sycl_kernel.cc"],
    hdrs = ["sycl_kernel.h"],
    deps = [
        ":sycl_types_header",
        "//xla/stream_executor/sycl:itex_gpu_runtime_imp",
        "@xla//xla/stream_executor:event",
        "@xla//xla/stream_executor:stream_executor_pimpl_header",
        "@xla//xla/stream_executor/platform",
    ],
)

cc_library(
    name = "sycl_event",
    srcs = ["sycl_event.cc"],
    hdrs = ["sycl_event.h"],
    deps = [
        ":sycl_executor_header",
        ":sycl_stream",
        "@xla//xla/stream_executor:stream_executor_headers",
    ],
)

cc_library(
    name = "sycl_stream",
    srcs = ["sycl_stream.cc"],
    hdrs = ["sycl_stream.h"],
    deps = [
        ":sycl_executor_header",
        "@xla//xla/stream_executor:stream_executor_headers",
        "@xla//xla/stream_executor/platform",
    ],
)

cc_library(
    name = "sycl_timer",
    srcs = ["sycl_timer.cc"],
    hdrs = ["sycl_timer.h"],
    deps = [
        ":sycl_executor_header",
        ":sycl_stream",
        "@xla//xla/stream_executor:stream_executor_headers",
    ],
)

cc_library(
    name = "sycl_executor",
    srcs = ["sycl_executor.cc"],
    hdrs = ["sycl_executor.h"],
    deps = [
        ":sycl_event",
        ":sycl_kernel",
        ":sycl_platform_id",
        ":sycl_stream",
        ":sycl_timer",
        "//xla/stream_executor/sycl:gpu_pool_allocator",
        "@com_google_absl//absl/strings",
        "@xla//xla/stream_executor:event",
        "@xla//xla/stream_executor:stream_executor_internal",
        "@xla//xla/stream_executor:stream_executor_pimpl_header",
        "@xla//xla/stream_executor:timer",
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
