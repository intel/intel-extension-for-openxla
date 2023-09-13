load("//xla:xla.bzl", "xetla_library")

# List all kernels here.
xetla_library(
    name = "gemm_kernel",
    srcs = [
        "gemm.cc",
    ],
    hdrs = [
        "XEGEMM.h",
        "gemm.h",
        "hgemm_impl.h",
        "epilogue_impl.h",
    ],
    copts = [
        "-Wall",
        "-Wno-c++11-narrowing",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/service/gpu:matrix_descriptor",
        "//xla/stream_executor/sycl:sycl_executor",
        "@xetla//:xetla_header",
    ],
)
