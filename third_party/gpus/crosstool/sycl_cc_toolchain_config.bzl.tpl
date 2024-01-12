"""cc_toolchain_config rule for configuring DPC++ toolchains on Linux."""

load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "tool_path",
    "variable_with_value",
    "with_feature_set",
)
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

all_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.clif_match,
    ACTION_NAMES.lto_backend,
]

all_cpp_compile_actions = [
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.clif_match,
]

preprocessor_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.clif_match,
]

codegen_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.lto_backend,
]

all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

lto_index_actions = [
    ACTION_NAMES.lto_index_for_executable,
    ACTION_NAMES.lto_index_for_dynamic_library,
    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
]


def _impl(ctx):
    tool_paths = [
        tool_path(name = "gcc", path = ctx.attr.host_compiler_path),
        tool_path(name = "ar", path = ctx.attr.ar_path),
        # tool_path(name = "ar", path = ctx.attr.host_compiler_prefix + "/ar"),
        tool_path(name = "compat-ld", path = ctx.attr.host_compiler_prefix + "/ld"),
        tool_path(name = "cpp", path = ctx.attr.host_compiler_prefix + "/cpp"),
        tool_path(name = "dwp", path = ctx.attr.host_compiler_prefix + "/dwp"),
        tool_path(name = "gcov", path = ctx.attr.host_compiler_prefix + "/gcov"),
        tool_path(name = "ld", path = ctx.attr.host_compiler_prefix + "/ld"),
        tool_path(name = "nm", path = ctx.attr.host_compiler_prefix + "/nm"),
        tool_path(name = "objcopy", path = ctx.attr.host_compiler_prefix + "/objcopy"),
        tool_path(name = "objdump", path = ctx.attr.host_compiler_prefix + "/objdump"),
        tool_path(name = "strip", path = ctx.attr.host_compiler_prefix + "/strip"),
    ]

    action_configs = []

    pic_feature = feature(
        name = "pic",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(flags = ["-fPIC"], expand_if_available = "pic"),
                    flag_group(
                        flags = ["-fPIE"],
                        expand_if_not_available = "pic",
                    ),
                ],
            ),
        ],
    )

    dbg_feature = feature(
        name = "dbg",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-g"])],
            ),
        ],
        implies = ["common"],
    )

    opt_feature = feature(
        name = "opt",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = ["-O3", "-ffunction-sections", "-fdata-sections"],
                    ),
                ],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [flag_group(flags = ["-Wl,--gc-sections"])],
            ),
        ],
        implies = ["common", "disable-assertions"],
    )

    unfiltered_compile_flags_feature = feature(
        name = "unfiltered_compile_flags",
        flag_sets = ([
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ctx.attr.unfiltered_compile_flags,
                    ),
                ],
            ),
        ] if ctx.attr.unfiltered_compile_flags else []),
    )

    determinism_feature = feature(
        name = "determinism",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Wno-builtin-macro-redefined",
                            "-D__DATE__=\"redacted\"",
                            "-D__TIMESTAMP__=\"redacted\"",
                            "-D__TIME__=\"redacted\"",
                            "-no-canonical-prefixes",
                        ],
                    ),
                ],
            ),
        ],
    )

    supports_pic_feature = feature(name = "supports_pic", enabled = True)

    hardening_feature = feature(
        name = "hardening",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-U_FORTIFY_SOURCE",
                            "-D_FORTIFY_SOURCE=1",
                            "-fstack-protector",
                        ],
                    ),
                ],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = ["-Wl,-z,relro,-z,now"])],
            ),
            flag_set(
                actions = [ACTION_NAMES.cpp_link_executable],
                flag_groups = [flag_group(flags = ["-pie", "-Wl,-z,relro,-z,now"])],
            ),
        ],
    )

    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)

    warnings_feature = feature(
        name = "warnings",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = ["-Wall"],
                    ),
                ],
            ),
        ],
    )

    disable_assertions_feature = feature(
        name = "disable-assertions",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-DNDEBUG"])],
            ),
        ],
    )

    linker_bin_path_feature = feature(
        name = "linker-bin-path",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["-B" + ctx.attr.linker_bin_path])],
            ),
        ],
    )

    frame_pointer_feature = feature(
        name = "frame-pointer",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-fno-omit-frame-pointer"])],
            ),
        ],
    )

    build_id_feature = feature(
        name = "build-id",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,--build-id=md5", "-Wl,--hash-style=gnu"],
                    ),
                ],
            ),
        ],
    )

    stdlib_feature = feature(
        name = "stdlib",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["-lstdc++", "-B/usr/bin"])],
            ),
        ],
    )

    alwayslink_feature = feature(
        name = "alwayslink",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = [
                    "-link_stage",
                    "-rdynamic",
                    "-fPIC",
                    "-no-canonical-prefixes",
                    "-Wl,-no-as-needed",
                    "-Wl,-z,relro,-z,now",
                    "-Wl,--build-id=md5",
                    "-Wl,--hash-style=gnu",
                ])],
            ),
        ],
    )

    mkl_link_feature = feature(
        name = "mkl_link",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = [
                    "-L%{MKL_PATH}/lib/",
                    "-Wl,-no-as-needed",
                    "-lmkl_intel_ilp64",
                    "-lmkl_sequential",
                    "-lmkl_core",
                    "-lmkl_sycl_blas",
                    "-lmkl_sycl_lapack",
                    "-lmkl_sycl_sparse",
                    "-lmkl_sycl_dft",
                    "-lmkl_sycl_vm",
                    "-lmkl_sycl_rng",
                    "-lmkl_sycl_stats",
                    "-lmkl_sycl_data_fitting",
                ])],
            ),
        ],
    )

    host_link_feature = feature(
        name = "host_link",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = [
                    "-Wl,-Bstatic,-lsvml,-lirng,-limf,-lirc,-lirc_s,-Bdynamic"
                ])],
            ),
        ],
    )

    no_canonical_prefixes_feature = feature(
        name = "no-canonical-prefixes",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-no-canonical-prefixes",
                        ],
                    ),
                ],
            ),
        ],
    )

    sycl_compiler_feature = feature(
        name = "sycl_feature",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_compile,
                ],
                flag_groups = [
                    flag_group(flags = [
                        "-fPIC",
                        "-DDNNL_USE_DPCPP_USM=1",
                        "-DDNNL_WITH_LEVEL_ZERO=1",
                        "-DNGEN_NO_OP_NAMES=1",
                        "-DNGEN_CPP11=1",
                        "-DNGEN_SAFE=1",
                        "-DNGEN_NEO_INTERFACE=1",
                        "-DDNNL_X64=1",
                        "-DEIGEN_HAS_C99_MATH=1",
                        "-DEIGEN_HAS_CXX11_MATH=1",
                        "-Wno-unused-variable",
                        "-Wno-unused-const-variable",
                    ]),
                ],
            ),
        ],
    )

    common_feature = feature(
        name = "common",
        implies = [
            "stdlib",
            "sycl_feature",
            "determinism",
            "alwayslink",
            "hardening",
            "warnings",
            "frame-pointer",
            "build-id",
            "no-canonical-prefixes",
            "linker-bin-path",
        ],
    )
    features = [
        mkl_link_feature,
        host_link_feature,
        sycl_compiler_feature,
        stdlib_feature,
        determinism_feature,
        alwayslink_feature,
        pic_feature,
        hardening_feature,
        warnings_feature,
        frame_pointer_feature,
        build_id_feature,
        no_canonical_prefixes_feature,
        disable_assertions_feature,
        linker_bin_path_feature,
        common_feature,
        opt_feature,
        dbg_feature,
        supports_dynamic_linker_feature,
        supports_pic_feature,
        unfiltered_compile_flags_feature,
    ]
    sys_inc = [
        "/usr/lib",
        "/usr/lib64",
        # for GPU kernel's header file
        "%{TMP_DIRECTORY}",
    ]
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = action_configs,
        artifact_name_patterns = [],
        cxx_builtin_include_directories = sys_inc + ctx.attr.cxx_builtin_include_directories,
        toolchain_identifier = ctx.attr.toolchain_identifier,
        host_system_name = ctx.attr.host_system_name,
        target_system_name = ctx.attr.target_system_name,
        target_cpu = ctx.attr.cpu,
        target_libc = ctx.attr.target_libc,
        compiler = ctx.attr.compiler,
        abi_version = ctx.attr.abi_version,
        abi_libc_version = ctx.attr.abi_libc_version,
        tool_paths = tool_paths,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory = True),
        "compiler": attr.string(mandatory = True),
        "toolchain_identifier": attr.string(mandatory = True),
        "host_system_name": attr.string(mandatory = True),
        "target_system_name": attr.string(mandatory = True),
        "target_libc": attr.string(mandatory = True),
        "abi_version": attr.string(mandatory = True),
        "abi_libc_version": attr.string(mandatory = True),
        "cxx_builtin_include_directories": attr.string_list(),
        "compile_flags": attr.string_list(),
        "dbg_compile_flags": attr.string_list(),
        "opt_compile_flags": attr.string_list(),
        "cxx_flags": attr.string_list(),
        "link_flags": attr.string_list(),
        "link_libs": attr.string_list(),
        "opt_link_flags": attr.string_list(),
        "unfiltered_compile_flags": attr.string_list(),
        "coverage_compile_flags": attr.string_list(),
        "coverage_link_flags": attr.string_list(),
        "host_compiler_path": attr.string(),
        "host_compiler_prefix": attr.string(),
        "linker_bin_path": attr.string(),
        "ar_path": attr.string(),
    },
    provides = [CcToolchainConfigInfo],
)
