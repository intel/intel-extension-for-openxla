load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//lib:selects.bzl", "selects")

exports_files(["LICENSE"])

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

config_setting(
    name = "build_with_onednn",
    define_values = {
        "build_with_onednn": "true",
    },
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)

py_binary(
    name = "gen_gpu_kernel_list",
    srcs = ["gen_gpu_kernel_list.py"],
    visibility = [
        "@onednn_gpu//:__subpackages__",
    ],
)

py_binary(
    name = "gen_onednn_version",
    srcs = ["gen_onednn_version.py"],
    visibility = [
        "@onednn_cpu//:__subpackages__",
        "@onednn_gpu//:__subpackages__",
    ],
)
