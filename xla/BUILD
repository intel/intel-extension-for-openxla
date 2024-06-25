cc_binary(
    name = "pjrt_plugin_xpu.so",
    linkopts = ["-Wl,-rpath,$$ORIGIN/../intel_extension_for_openxla/service/gpu"],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        "@xla//xla/pjrt/c:pjrt_c_api_gpu",
        "@xla//xla/service:gpu_plugin",
        "//xla/stream_executor:sycl_platform",
    ],
)
