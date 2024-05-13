package(default_visibility = ["//visibility:public"])

load(":platform.bzl", "sycl_library_path")
load("@local_config_sycl//sycl:build_defs.bzl", "if_sycl")

config_setting(
    name = "using_sycl",
    values = {
        "define": "using_sycl=true",
    },
)

config_setting(
    name = "using_xetla",
    values = {
        "define": "using_xetla=true",
    },
)
