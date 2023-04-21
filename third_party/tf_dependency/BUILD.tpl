package(default_visibility = ["//visibility:public"])

cc_library(
    name = "jax_internal",
    srcs = ["%{JAX_SHARED_LIBRARY_NAME}"],
    visibility = ["//visibility:public"],
)

%{JAX_SHARED_LIBRARY_GENRULE}
