package(default_visibility = ["//visibility:public"])

cc_library(
    name = "llvm_gpu_backend",
    srcs = [
        "gpu_backend_lib.cc",
        "utils.cc",
    ],
    hdrs = [
        "gpu_backend_lib.h",
        "utils.h",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:ObjCARC",  # buildcleaner: keep
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm_spir//:llvm_spir_translator",
        "@xla//xla:status_macros",
        "@xla//xla:statusor",
        "@xla//xla:types",
        "@xla//xla:util",
        "@xla//xla/service:dump",
        "@xla//xla/service:hlo_module_config",
        "@xla//xla/service/llvm_ir:llvm_command_line_options",
        "@xla//xla/service/llvm_ir:llvm_type_conversion_util",
        "@xla//xla/service/llvm_ir:llvm_util",
    ],
)
