load("//xla:xla.bzl", "xpu_library")
load("//xla:xla.bzl", "xetla_library")
load(
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "matrix_descriptor",
    hdrs = ["matrix_descriptor.h"],
    deps = [
        "//xla/stream_executor/sycl:sycl_executor",
        "@xla//xla/stream_executor:stream_executor_headers",
    ],
)

cc_library(
    name = "sycl_custom_call",
    srcs = [
        "sycl_custom_call.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/service:onednn_util",
        "//xla/service/gpu:sycl_onednn",
        "@xla//xla/ffi",
        "@xla//xla/ffi:ffi_api",
        "@xla//xla/stream_executor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

xetla_library(
    name = "onednn_matmul_utils",
    srcs = ["onednn_matmul_utils.cc"],
    hdrs = ["onednn_matmul_utils.h"],
    deps = [
        ":scratch_allocator",
        "//xla/service:onednn_util",
        "//xla/service/gpu/xetla/gemm:gemm_kernel",
        "@xla//xla/service/gpu:matmul_utils",
    ],
)

cc_library(
    name = "gemm_impl_picker",
    srcs = ["gemm_impl_picker.cc",],
    hdrs = ["gemm_impl_picker.h"],
    deps = [
        ":sycl_onednn",
        "//xla/stream_executor/sycl:hw_info",
        "@com_google_absl//absl/algorithm:container",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/util:env_var",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_pass",
        "@xla//xla/service/gpu:autotuner_util",
        "@xla//xla/service/gpu:cublas_cudnn",
        "@xla//xla/stream_executor",
        "@xla//xla/stream_executor:device_description",
        "@xla//xla/stream_executor:device_memory_allocator",
        "@xla//xla/stream_executor/gpu:gpu_stream",
        "@xla//xla/stream_executor/gpu:gpu_timer",
        "@xla//xla/service/gpu:ir_emission_utils",
        "@xla//xla/service/gpu:matmul_utils",
        "@xla//xla/service/gpu:stream_executor_util",
        "@xla//xla:util",
    ]
)

xpu_library(
    name = "ccl_collective_thunks",
    srcs = [
        "ccl_all_to_all_thunk.cc",
        "ccl_collective_permute_thunk.cc",
        "ccl_collective_thunk.cc",
        "ccl_p2p_thunk_common.cc",
        "ccl_all_gather_thunk.cc",
        "ccl_all_reduce_thunk.cc",
    ],
    hdrs = [
        "ccl_all_to_all_thunk.h",
        "ccl_collective_permute_thunk.h",
        "ccl_collective_thunk.h",
        "ccl_p2p_thunk_common.h",
        "ccl_all_gather_thunk.h",
        "ccl_all_reduce_thunk.h",
    ],
    deps = [
        "@xla//xla/service/gpu:backend_configs_cc",
        "@xla//xla/service/gpu:buffer_allocations",
        "@xla//xla/service/gpu:gpu_executable_run_options",
        "@xla//xla/service/gpu:ir_emission_utils",
        "@xla//xla/service/gpu:nccl_api",
        "@xla//xla/service/gpu:nccl_clique",
        "@xla//xla/service/gpu:nccl_clique_key",
        "@xla//xla/service/gpu:thunk",
        "@xla//xla:shape_util",
        "@xla//xla:status",
        "@xla//xla:status_macros",
        "@xla//xla:statusor",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/mlir_hlo:lhlo",
        "@xla//xla/mlir_hlo:lhlo_gpu",
        "@xla//xla/service:buffer_assignment",
        "@xla//xla/service:collective_ops_utils",
        "@xla//xla/service:computation_placer",
        "@xla//xla/service:global_device_id",
        "@xla//xla/service:hlo_parser",
        "@xla//xla/service:rendezvous",
        "@xla//xla/service/llvm_ir:llvm_util",
        "@xla//xla/stream_executor",
        "@xla//xla/stream_executor/gpu:gpu_activation_header",
        "@xla//xla/stream_executor/gpu:gpu_driver_header",
        "@xla//xla/stream_executor/gpu:gpu_stream",
        "@xla//xla/stream_executor/gpu:gpu_types_header",
        "@xla//xla/translate/hlo_to_mhlo:hlo_utils",
        "@xla//xla/translate/mhlo_to_hlo:attribute_exporter",
        "@xla//xla/translate/mhlo_to_hlo:type_to_shape",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:statusor",
    ],
)

xpu_library(
    name = "utils",
    hdrs = ["utils.h"],
)

xpu_library(
    name = "ccl_ops",
    srcs = [
        "ccl_ops.cc",
    ],
    hdrs = [
        "ccl_ops.h",
    ],

    deps = [
        "//xla/service/gpu:utils",
        "//xla/stream_executor/sycl:sycl_driver",
        "//xla/stream_executor/sycl:sycl_gpu_runtime",
        "@xla//xla/service:collective_ops_utils",
        "@xla//xla/stream_executor/gpu:gpu_types_header",
    ],
)

cc_import(
    name = "sycl_onednn",
    hdrs = [
        "sycl_onednn.h",
        "onednn_gpu_conv_runner.h",
        "onednn_matmul_utils.h",
    ],
    shared_library = ":sycl_onednn.so",
    visibility = ["//visibility:public"],
    deps = [
        ":scratch_allocator",
        "@xla//xla/service/gpu:gpu_conv_runner",
        "@xla//xla/service/gpu:thunk",
        "@xla//xla/service/gpu:matmul_utils",
    ],
)

cc_binary(
    name = "sycl_onednn.so",
    srcs = [
        "sycl_onednn.cc",
        "sycl_onednn.h",
    ],
    linkshared = True,
    deps = [
        ":onednn_gpu_conv_runner",
        ":onednn_matmul_utils",
    ],
)

cc_library(
    name = "onednn_gpu_conv_runner",
    srcs = [
        "onednn_gpu_conv_runner.cc",
        "onednn_gpu_conv_runner.h",
    ],
    deps = [
        ":scratch_allocator",
        "//xla/service:onednn_util",
        "@xla//xla/ffi",
        "@xla//xla/ffi:ffi_api",
        "@xla//xla/service/gpu:gpu_conv_runner",
        "@xla//xla/service/gpu:stream_executor_util",
        "@xla//xla/service/gpu:thunk",
    ],
)

cc_library(
    name = "scratch_allocator",
    srcs = ["scratch_allocator.cc"],
    hdrs = ["scratch_allocator.h"],
    deps = [
        "@xla//xla/stream_executor:scratch_allocator",
    ],
)

cc_library(
    name = "fused_mha_rewriter",
    srcs = ["fused_mha_rewriter.cc"],
    hdrs = ["fused_mha_rewriter.h"],
    deps = [
        "//xla/stream_executor/sycl:hw_info",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
        "@xla//xla:comparison_util",
        "@xla//xla:literal_util",
        "@xla//xla:permutation_util",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_creation_utils",
        "@xla//xla/service:hlo_pass",
        "@xla//xla/service:pattern_matcher",
        "@xla//xla/service/gpu:backend_configs_cc",
        "@xla//xla/service/gpu:cublas_cudnn",
        "@xla//xla/service/gpu:ir_emission_utils",
        "@xla//xla/service/gpu:matmul_utils",
    ],
)

cc_library(
    name = "xetla_gpu_fused_mha_runner",
    srcs = ["xetla_gpu_fused_mha_runner.cc"],
    hdrs = ["xetla_gpu_fused_mha_runner.h"],
    deps = [
        "//xla/service/gpu/xetla/sdp:sdp_backward_kernel",
        "//xla/service/gpu/xetla/sdp:sdp_forward_kernel",
        "//xla/stream_executor/sycl:sycl_executor",
        "@com_google_absl//absl/strings",
        "@xla//xla:shape_util",
        "@xla//xla:status",
        "@xla//xla:status_macros",
        "@xla//xla:statusor",
        "@xla//xla:types",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service/gpu:backend_configs_cc",
        "@xla//xla/service/gpu:cublas_cudnn",
        "@xla//xla/service/gpu:gpu_fused_mha_runner",
        "@xla//xla/service/gpu:stream_executor_util",
        "@xla//xla/stream_executor",
        "@xla//xla/stream_executor:dnn",
        "@xla//xla/stream_executor:lazy_op_runner",
    ],
)

cc_library(
    name = "dot_expand_dims",
    srcs = ["dot_expand_dims.cc"],
    hdrs = ["dot_expand_dims.h"],
    deps = [
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
        "@xla//xla:comparison_util",
        "@xla//xla:literal_util",
        "@xla//xla:permutation_util",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_creation_utils",
        "@xla//xla/service:hlo_pass",
        "@xla//xla/service:pattern_matcher",
        "@xla//xla/service/gpu:ir_emission_utils",
        "@xla//xla/service/gpu:matmul_utils",
    ],
)

cc_library(
    name = "redundant_convert_mover",
    srcs = ["redundant_convert_mover.cc"],
    hdrs = ["redundant_convert_mover.h"],
    deps = [
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:statusor",
        "@xla//xla:comparison_util",
        "@xla//xla:literal_util",
        "@xla//xla:permutation_util",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_creation_utils",
        "@xla//xla/service:hlo_pass",
        "@xla//xla/service:pattern_matcher",
    ],
)