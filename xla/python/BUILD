load("@tsl//tsl:tsl.default.bzl", "tsl_pybind_extension")

cc_library(
    name = "py_client_gpu",
    srcs = ["py_client_gpu.cc"],
    hdrs = [
        "py_client_gpu.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor/sycl:sycl_executor",
        "@xla//xla/python:py_client",
    ],
    alwayslink = 1,
)

tsl_pybind_extension(
    name = "xpu_plugin_extension",
    srcs = ["xpu_plugin_extension.cc"],
    deps = [
        "@pybind11",
        "@xla//xla:status",
        "@xla//xla:util",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_helpers",
        "@xla//xla/python:status_casters",
    ],
)
