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
