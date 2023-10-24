package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_sycl",
    values = {
        "define": "using_sycl=true",
    },
)
