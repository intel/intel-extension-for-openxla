load("@xla//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")

tsl_pybind_extension(
    name = "xpu_plugin_extension",
    srcs = ["xpu_plugin_extension.cc"],
    deps = [
        "@nanobind",
        "@xla//xla:status",
        "@xla//xla:util",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_helpers",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/python:py_client_gpu",
        "@xla//xla/tsl/python/lib/core:numpy",
    ],
)
