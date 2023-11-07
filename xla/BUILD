cc_binary(
    name = "pjrt_plugin_xpu.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        "//xla/pjrt:pjrt_c_api_xpu",
        "//xla/python:py_client_gpu",
    ],
)
