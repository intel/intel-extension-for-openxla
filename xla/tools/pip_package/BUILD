# Description:
#  Tools for building the TensorFlow pip package.

package(default_visibility = ["//visibility:private"])

load("//xla:xla.bzl", "transitive_hdrs")

COMMON_PIP_DEPS = [
    "MANIFEST.in",
    "README.md",
    "xla_setup.py",
    "//xla:pjrt_plugin_xpu.so",
]

py_binary(
    name = "simple_console",
    srcs = ["simple_console.py"],
    srcs_version = "PY2AND3",
    deps = [],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = ["simple_console"] + COMMON_PIP_DEPS,
)
