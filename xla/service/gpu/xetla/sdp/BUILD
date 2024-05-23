load("//xla:xla.bzl", "xetla_library")

# List all kernels here.
xetla_library(
    name = "sdp_forward_kernel",
    srcs = [
        "sdp_forward.cc",
    ],
    hdrs = [
        "sdp_forward.h",
        "fmha_forward.h",
        "fmha_policy.h",
        "fmha_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@xetla//:xetla_header",
        "@tsl//tsl/platform:logging",
    ],
)

xetla_library(
    name = "sdp_backward_kernel",
    srcs = [
        "sdp_backward.cc",
    ],
    hdrs = [
        "sdp_backward.h",
        "fmha_backward.h",
        "fmha_policy.h",
        "fmha_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@xetla//:xetla_header",
        "@tsl//tsl/platform:logging",
    ],
)

