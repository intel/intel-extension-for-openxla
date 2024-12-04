load("//third_party/gpus:sycl_configure.bzl", "sycl_configure")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

def workspace(path_prefix = "", tf_repo_name = ""):
    sycl_configure(name = "local_config_sycl")

    new_git_repository(
        name = "onednn_gpu",
        # v3.6.1
        commit = "e72f65d70e36f552f902d35614aef7aa54f3c796",
        remote = "https://github.com/oneapi-src/oneDNN.git",
        build_file = "//third_party/onednn:onednn_gpu.BUILD",
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    git_repository(
        name = "spir_headers",
        commit = "4f7b471f1a66b6d06462cd4ba57628cc0cd087d7",
        remote = "https://github.com/KhronosGroup/SPIRV-Headers.git",
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    new_git_repository(
        name = "llvm_spir",
        commit = "98729ef13ca970b98a149c18318746df8d921f20",
        remote = "https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git",
        build_file = "//third_party/llvm_spir:llvm_spir.BUILD",
        verbose = True,
        patches = ["//third_party/llvm_spir:llvm_spir.patch"],
        patch_args = ["-p1"],
    )

    new_git_repository(
        name = "xetla",
        patch_args = ["-p1"],
        patches = ["//third_party/xetla:xetla.patch"],
        # v0.3.7.2
        commit = "ae46a690bac364a93437e248418636c2a8423134",
        remote = "https://github.com/intel/xetla.git",
        verbose = True,
        build_file = "//third_party/xetla:BUILD",
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )
