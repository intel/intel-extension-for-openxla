load("//xla:xla.bzl", "xetla_library")

# List all kernels here.
xetla_library(
    name = "sdp_kernel",
    srcs = [
        "sdp.cc",
    ],
    hdrs = [
        "fmha_forward.h",
        "fmha_backward.h",
        "fmha_policy.h",
        "fmha_utils.h",
        "sdp.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@xetla//:xetla_header",
        "@tsl//tsl/platform:logging",
    ],
)
