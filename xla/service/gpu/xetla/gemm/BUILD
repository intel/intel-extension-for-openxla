load("//xla:xla.bzl", "xetla_library")

# List all kernels here.
xetla_library(
    name = "gemm_kernel",
    srcs = [
        "gemm.cc",
    ],
    hdrs = [
        "gemm.h",
        "hgemm_splitk.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@xetla//:xetla_header",
    ],
)
