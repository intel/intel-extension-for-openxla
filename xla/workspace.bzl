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
        commit = "1feaf4414eb2b353764d01d88f8aa4bcc67b60db",
        remote = "https://github.com/KhronosGroup/SPIRV-Headers.git",
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    new_git_repository(
        name = "llvm_spir",
        commit = "34487407835afc1a8b7ad277935bf950c5adecde",
        remote = "https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git",
        build_file = "//third_party/llvm_spir:llvm_spir.BUILD",
        verbose = True,
        patches = ["//third_party/llvm_spir:llvm_spir.patch"],
        patch_args = ["-p1"],
    )

    new_git_repository(
        name = "xetla",
        # v0.3.1
        commit = "2c29086eba7bf8369c49a6da8dd9b2912e954d20",
        remote = "https://github.com/intel/xetla",
        verbose = True,
        build_file = "//third_party/xetla:BUILD",
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )
