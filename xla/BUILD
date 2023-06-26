load("//xla:xla.bzl", "xpu_binary")

xpu_binary(
    name = "libitex_xla_extension.so",
    linkopts = ["-Wl,-rpath,$$ORIGIN/../jaxlib"],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            "//xla/pjrt:xpu_pjrt_client",
            "@local_config_xla//:jax_internal",
        ],
        "//conditions:default": [],
    }),
)
