load("//xla:xla.bzl", "itex_xetla_library")

# List all kernels here.
itex_xetla_library(
    name = "mha_op",
    srcs = [
        "fmha_forward.h",
        "fmha_policy.h",
        "fmha_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@xetla//:xetla_header",
    ],
)