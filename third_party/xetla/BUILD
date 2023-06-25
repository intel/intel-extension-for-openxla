package(default_visibility = ["//visibility:public"])

cc_library(
    name = "xetla_header",
    hdrs = glob(
        include = [
            "include/util/*.h",
            "include/tile/*.h",
            "include/reduction/*.h",
            "include/mha_core_attention/*.h",
            "include/layer_norm/*.h",
            "include/gemm/*.h",
            "include/data_transformer/*.h",
            "include/core/*.h",
            "include/brgemm/*.h",            
        ],
    ),
    includes = ["include"],
    strip_include_prefix = "include",
)