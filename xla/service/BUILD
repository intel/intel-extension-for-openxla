load("//third_party/onednn:build_defs.bzl", "onednn_deps")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "onednn_util",
    hdrs = ["onednn_util.h"],
    deps = [
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:types",
    ] + onednn_deps(),
)
