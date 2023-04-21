load("//xla:xla.bzl", "itex_xpu_binary", "itex_xpu_library")

itex_xpu_binary(
    name = "libitex_xla_extension.so",
    linkopts = ["-Wl,-rpath,$$ORIGIN/../jaxlib"],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = select({
        "@local_config_dpcpp//dpcpp:using_dpcpp": [
            "//xla/pjrt:xpu_pjrt_client",
            "@local_config_tf//:jax_internal",
        ],
        "//conditions:default": [],
    }),
)
