load("//third_party/build_option:dpcpp_configure.bzl", "dpcpp_configure")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

def workspace(path_prefix = "", tf_repo_name = ""):
    dpcpp_configure(name = "local_config_dpcpp")

    new_git_repository(
        name = "onednn_gpu",
        # rls-v3.2
        commit = "f88241a",
        remote = "https://github.com/oneapi-src/oneDNN.git",
        build_file = "//third_party/onednn:onednn_gpu.BUILD",
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    git_repository(
        name = "spir_headers",
        commit = "29ba2493125effc581532518add689613cebfec7",
        remote = "https://github.com/KhronosGroup/SPIRV-Headers.git",
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    new_git_repository(
        name = "llvm_spir",
        commit = "0ef30a3dfeec433b6d4de4e9cc2a6817032b93e3",
        remote = "https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git",
        build_file = "//third_party/llvm_spir:llvm_spir.BUILD",
        verbose = True,
        patches = ["//third_party/llvm_spir:llvm_spir.patch"],
        patch_args = ["-p1"],
    )

    new_git_repository(
        name = "xetla",
        # v0.3.1
        commit = "b18d66e127e0db05c2a3ffed99792c361ce2b7b6",
        remote = "https://github.com/intel-innersource/libraries.gpu.xetla.git",
        verbose = True,
        build_file = "//third_party/xetla:BUILD",
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ], 
    )

    # _itex_bind()
