package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_sycl",
    values = {
        "define": "using_sycl=true",
    },
)

cc_library(
    name = "sycl_headers",
    hdrs = [
        %{sycl_headers}
    ],
    includes = [
        ".",
        "sycl/include",
        "sycl/include/sycl",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl",
    srcs = [
        "sycl/lib/%{mkl_intel_ilp64_lib}",
        "sycl/lib/%{mkl_sequential_lib}",
        "sycl/lib/%{mkl_core_lib}",
        %{mkl_sycl_libs}
    ],
    data = [
        "sycl/lib/%{mkl_intel_ilp64_lib}",
        "sycl/lib/%{mkl_sequential_lib}",
        "sycl/lib/%{mkl_core_lib}",
        %{mkl_sycl_libs}
    ],
    includes = [
        ".",
        "sycl/include",
    ],
    linkopts = ["-Wl,-Bstatic,-lsvml,-lirng,-limf,-lirc,-lirc_s,-Bdynamic"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "level_zero",
    srcs = [
        %{level_zero_libs}
    ],
    data = [
        %{level_zero_libs}
    ],
    hdrs = [
        %{level_zero_headers}
    ],
    includes = [
        ".",
        "level_zero/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

%{copy_rules}