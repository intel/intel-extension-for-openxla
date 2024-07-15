load("//xla:xla.bzl", "xetla_library")

xetla_library(
    name = "gemm_common", 
    hdrs = [
        "gemm_common.h"
    ],
    copts = [
        "-Wall",
        "-Wno-c++11-narrowing",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/service/gpu:matrix_descriptor",
    ],
)

xetla_library(
    name = "gemm_dispatch", 
    hdrs = [
        "gemm_dispatch.h",
        "hgemm_impl.h",
        "epilogue_impl.h",
    ],
    copts = [
        "-Wall",
        "-Wno-c++11-narrowing",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":gemm_common",
        "//xla/service/gpu:matrix_descriptor",
        "//xla/stream_executor/sycl:sycl_executor",
        "@xetla//:xetla_header",
        "@com_google_absl//absl/strings",
    ],
)

xetla_library(
    name = "dispatch_row_major",
    srcs = [
        "dispatch_row_major.cc",
    ],
    hdrs = [
        "dispatch_row_major.h",
    ],
    copts = [
        "-Wall",
        "-Wno-c++11-narrowing",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":gemm_dispatch",
        "//xla/stream_executor/sycl:sycl_executor",
    ],
)

xetla_library(
    name = "dispatch_col_major",
    srcs = [
        "dispatch_col_major.cc",
    ],
    hdrs = [
        "dispatch_col_major.h",
    ],
    copts = [
        "-Wall",
        "-Wno-c++11-narrowing",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":gemm_dispatch",
        "//xla/stream_executor/sycl:sycl_executor",
    ],
)

xetla_library(
    name = "gemm_kernel",
    srcs = [
        "gemm.cc",
    ],
    hdrs = [
        "gemm.h",
    ],
    copts = [
        "-Wall",
        "-Wno-c++11-narrowing",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":gemm_common",
        ":dispatch_row_major",
        ":dispatch_col_major",
        "//xla/service/gpu:matrix_descriptor",
        "//xla/stream_executor/sycl:sycl_executor",
        "@xetla//:xetla_header",
        "@com_google_absl//absl/strings",
    ],
)
