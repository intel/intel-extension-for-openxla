cc_binary(
    name = "pjrt_plugin_xpu.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            "//xla/pjrt:xpu_pjrt_client",
            "//xla/python:py_client_gpu",
        ],
        "//conditions:default": [],
    }),
)
