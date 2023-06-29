cc_binary(
    name = "libitex_xla_extension.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            "//xla/pjrt:xpu_pjrt_client",
        ],
        "//conditions:default": [],
    }),
)
