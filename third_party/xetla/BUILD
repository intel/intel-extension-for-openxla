package(default_visibility = ["//visibility:public"])

cc_library(
    name = "xetla_header",
    hdrs = glob(
        include = [
            "include/*.hpp",
            "include/common/*.hpp",
            "include/common/utils/*.hpp",
            "include/experimental/*.hpp",
            "include/group/*.hpp",
            "include/kernel/*.hpp",
            "include/subgroup/*.hpp",
        ],
    ),
    includes = ["include"],
    strip_include_prefix = "include",
)
