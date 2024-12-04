exports_files(["LICENSE"])

load(
    "@intel_extension_for_openxla//third_party:common.bzl",
    "template_rule",
)
load(
    "@intel_extension_for_openxla//third_party/onednn:onednn.bzl",
    "convert_cl_to_cpp",
    "convert_header_to_cpp",
    "gen_onednn_version",
)
load("@local_config_sycl//sycl:build_defs.bzl", "if_sycl_build_is_configured")

config_setting(
    name = "clang_linux_x86_64",
    values = {
        "cpu": "k8",
        "define": "using_clang=true",
    },
)

# TODO(itex): try better bazel usage in configuring strings with different options
_CMAKE_COMMON_LIST = {
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_NONE",
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_SEQ",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": if_sycl_build_is_configured("#define DNNL_GPU_RUNTIME DNNL_RUNTIME_DPCPP", "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_SYCL"),
    "#cmakedefine DNNL_SYCL_DPCPP": if_sycl_build_is_configured("#define DNNL_SYCL_DPCPP", "/* #undef DNNL_SYCL_DPCPP */"),
    "#cmakedefine DNNL_SYCL_COMPUTECPP": if_sycl_build_is_configured("/*#undef DNNL_SYCL_COMPUTECPP*/", "#define DNNL_SYCL_COMPUTECPP"),
    "#cmakedefine DNNL_WITH_LEVEL_ZERO": if_sycl_build_is_configured("/*#undef DNNL_WITH_LEVEL_ZERO*/", "/*#undef DNNL_WITH_LEVEL_ZERO*/"),
    "#cmakedefine DNNL_SYCL_CUDA": "/* #undef DNNL_SYCL_CUDA */",
    "#cmakedefine DNNL_SYCL_HIP": "/* #undef DNNL_SYCL_HIP */",
    "#cmakedefine DNNL_WITH_SYCL": if_sycl_build_is_configured("#define DNNL_WITH_SYCL", "/* #undef DNNL_WITH_SYCL */"),
    "#cmakedefine DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE": "#define DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE",
    "#cmakedefine DNNL_ENABLE_STACK_CHECKER": "#undef DNNL_ENABLE_STACK_CHECKER",
    "#cmakedefine DNNL_EXPERIMENTAL": "#define DNNL_EXPERIMENTAL",
    "#cmakedefine ONEDNN_BUILD_GRAPH": "#undef ONEDNN_BUILD_GRAPH",
    "#cmakedefine DNNL_SYCL_GENERIC": "#define DNNL_SYCL_GENERIC",
    "#cmakedefine DNNL_GPU_VENDOR DNNL_VENDOR_${DNNL_GPU_VENDOR}": "#define DNNL_GPU_VENDOR DNNL_VENDOR_INTEL",
    "#cmakedefine DNNL_DISABLE_GPU_REF_KERNELS": "#define DNNL_DISABLE_GPU_REF_KERNELS",
    "#cmakedefine DNNL_EXPERIMENTAL_LOGGING": "#undef DNNL_EXPERIMENTAL_LOGGING",
    "#cmakedefine01 BUILD_SDPA": "#define BUILD_SDPA 1",
    "#cmakedefine01 BUILD_TRAINING": "#define BUILD_TRAINING 1",
    "#cmakedefine01 BUILD_INFERENCE": "#define BUILD_INFERENCE 0",
    "#cmakedefine01 BUILD_PRIMITIVE_ALL": "#define BUILD_PRIMITIVE_ALL 1",
    "#cmakedefine01 BUILD_BATCH_NORMALIZATION": "#define BUILD_BATCH_NORMALIZATION 0",
    "#cmakedefine01 BUILD_BINARY": "#define BUILD_BINARY 0",
    "#cmakedefine01 BUILD_CONCAT": "#define BUILD_CONCAT 0",
    "#cmakedefine01 BUILD_CONVOLUTION": "#define BUILD_CONVOLUTION 0",
    "#cmakedefine01 BUILD_DECONVOLUTION": "#define BUILD_DECONVOLUTION 0",
    "#cmakedefine01 BUILD_ELTWISE": "#define BUILD_ELTWISE 0",
    "#cmakedefine01 BUILD_GROUP_NORMALIZATION": "#define BUILD_GROUP_NORMALIZATION 0",
    "#cmakedefine01 BUILD_INNER_PRODUCT": "#define BUILD_INNER_PRODUCT 0",
    "#cmakedefine01 BUILD_LAYER_NORMALIZATION": "#define BUILD_LAYER_NORMALIZATION 0",
    "#cmakedefine01 BUILD_LRN": "#define BUILD_LRN 0",
    "#cmakedefine01 BUILD_MATMUL": "#define BUILD_MATMUL 0",
    "#cmakedefine01 BUILD_POOLING": "#define BUILD_POOLING 0",
    "#cmakedefine01 BUILD_PRELU": "#define BUILD_PRELU 0",
    "#cmakedefine01 BUILD_REDUCTION": "#define BUILD_REDUCTION 0",
    "#cmakedefine01 BUILD_REORDER": "#define BUILD_REORDER 0",
    "#cmakedefine01 BUILD_RESAMPLING": "#define BUILD_RESAMPLING 0",
    "#cmakedefine01 BUILD_RNN": "#define BUILD_RNN 0",
    "#cmakedefine01 BUILD_SHUFFLE": "#define BUILD_SHUFFLE 0",
    "#cmakedefine01 BUILD_SOFTMAX": "#define BUILD_SOFTMAX 0",
    "#cmakedefine01 BUILD_SUM": "#define BUILD_SUM 0",
    "#cmakedefine01 BUILD_PRIMITIVE_CPU_ISA_ALL": "#define BUILD_PRIMITIVE_CPU_ISA_ALL 1",
    "#cmakedefine01 BUILD_SSE41": "#define BUILD_SSE41 0",
    "#cmakedefine01 BUILD_AVX2": "#define BUILD_AVX2 0",
    "#cmakedefine01 BUILD_AVX512": "#define BUILD_AVX512 0",
    "#cmakedefine01 BUILD_AMX": "#define BUILD_AMX 0",
    "#cmakedefine01 BUILD_PRIMITIVE_GPU_ISA_ALL": "#define BUILD_PRIMITIVE_GPU_ISA_ALL 1",
    "#cmakedefine01 BUILD_GEN9": "#define BUILD_GEN9 0",
    "#cmakedefine01 BUILD_GEN11": "#define BUILD_GEN11 0",
    "#cmakedefine01 BUILD_XELP": "#define BUILD_XELP 0",
    "#cmakedefine01 BUILD_XE2": "#define BUILD_XE2 0",
    "#cmakedefine01 BUILD_XEHPG": "#define BUILD_XEHPG 0",
    "#cmakedefine01 BUILD_XEHPC": "#define BUILD_XEHPC 0",
    "#cmakedefine01 BUILD_XEHP": "#define BUILD_XEHP 0",
    "#cmakedefine01 BUILD_GEMM_KERNELS_ALL": "#define BUILD_GEMM_KERNELS_ALL 1",
    "#cmakedefine01 BUILD_GEMM_KERNELS_NONE": "#define BUILD_GEMM_KERNELS_NONE 0",
    "#cmakedefine01 BUILD_GEMM_SSE41": "#define BUILD_GEMM_SSE41 0",
    "#cmakedefine01 BUILD_GEMM_AVX2": "#define BUILD_GEMM_AVX2 0",
    "#cmakedefine01 BUILD_GEMM_AVX512": "#define BUILD_GEMM_AVX512 0",
}

template_rule(
    name = "dnnl_config_h",
    src = "include/oneapi/dnnl/dnnl_config.h.in",
    out = "include/oneapi/dnnl/dnnl_config.h",
    substitutions = _CMAKE_COMMON_LIST,
)

convert_cl_to_cpp(
    name = "kernel_list_generator",
    src = "src/gpu/intel/ocl/ocl_kernel_list.cpp.in",
    cl_list = glob(["src/gpu/intel/ocl/**/*.cl"]),
)

convert_header_to_cpp(
    name = "header_generator",
    src = "src/gpu/intel/ocl/ocl_kernel_list.cpp.in",
    header_list = glob(["src/gpu/intel/**/*.h"]),
)

gen_onednn_version(
    name = "onednn_version_generator",
    header_in = "include/oneapi/dnnl/dnnl_version.h.in",
    header_out = "include/oneapi/dnnl/dnnl_version.h",
)

gen_onednn_version(
    name = "dnnl_version_hash_h",
    header_in = "include/oneapi/dnnl/dnnl_version_hash.h.in",
    header_out = "include/oneapi/dnnl/dnnl_version_hash.h",
)

filegroup(
    name = "onednn_src",
    srcs = glob(
        [
            "src/**/*.cpp",
            "src/**/*.hpp",
            "src/**/*.h",
            "src/**/*.c",
        ],
        exclude = [
            "src/cpu/**",
            "src/gpu/nvidia/*",
            "src/gpu/amd/*",
            "src/gpu/intel/jit/v2/conv/planner/*",
            "src/gpu/sycl/ref*",
            "src/graph/**",
        ],
    ) + [
        ":dnnl_config_h",
	":dnnl_version_hash_h",
        ":header_generator",
        ":kernel_list_generator",
        ":onednn_version_generator",
    ],
)

cc_library(
    name = "onednn_gpu",
    srcs = [":onednn_src"],
    hdrs = glob(
        [
            "include/*",
            "include/oneapi/dnnl/*",
        ],
        exclude = ["include/oneapi/dnnl/dnnl_graph*"],
    ),
    copts = [
        "-fexceptions",
        "-DDNNL_ENABLE_PRIMITIVE_CACHE",
        "-DDNNL_USE_DPCPP_USM=1",
        "-DDNNL_WITH_LEVEL_ZERO=1",
        "-DNGEN_NO_OP_NAMES=1",
        "-DNGEN_CPP11=1",
        "-DNGEN_SAFE=1",
        "-DNGEN_NEO_INTERFACE=1",
        #TODO(itex): for symbol collision, may be removed in produce version
        #"-fvisibility=hidden",
    ],
    includes = [
        "include",
        "include/oneapi",
        "include/oneapi/dnnl",
        "src",
        "src/common",
	"src/gpu/intel/jit/gemm/",
        "src/gpu/intel/jit/gemm/include/",
        "src/gpu/intel/jit/ngen/",
        "src/intel/ocl",
        "src/sycl",
    ],
    #nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_sycl//sycl:sycl_headers",
        "@intel_extension_for_openxla//xla/stream_executor/sycl:sycl_gpu_runtime"
    ],
)
