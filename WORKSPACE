workspace(name = "intel_extension_for_openxla")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("5.3.0")

load("//xla:workspace.bzl", "workspace")

workspace()

# To update XLA to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/openxla/xla/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.

http_archive(
    name = "xla",
    patch_args = ["-p1"],
    patches = ["//third_party:openxla.patch"],
    sha256 = "637acdfce5a406bd929bca052863be8200afbaac0daecbc547e9aff1967d7c00",
    strip_prefix = "xla-a5a30f7b34b69b7cada6deb4b39f77788e28369c",
    urls = [
        "https://github.com/openxla/xla/archive/a5a30f7b34b69b7cada6deb4b39f77788e28369c.tar.gz",
    ],
)

# For development, one often wants to make changes to the XLA repository as well
# as the JAX repository. You can override the pinned repository above with a
# local checkout by either:
# a) overriding the XLA repository by passing a flag like:
#    bazel build --bazel_options=--override_repository=xla=/path/to/xla
#    or
# b) by commenting out the http_archive above and uncommenting the following:
# local_repository(
#    name = "xla",
#    path = "/path/to/xla",
# )

load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

load(
    "@bazel_toolchains//repositories:repositories.bzl",
    bazel_toolchains_repositories = "repositories",
)

bazel_toolchains_repositories()
