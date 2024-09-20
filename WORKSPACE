workspace(name = "intel_extension_for_openxla")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("5.3.0")

# To update XLA to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/openxla/xla/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.
http_archive(
    name = "xla",
    patch_args = ["-p1"],
    patches = ["//third_party:openxla.patch"],
    sha256 = "083c7281a629647ab2cc32f054afec74893c33e75328783b8085c818f48235ff",
    strip_prefix = "xla-79fd5733f99b3c0948d7202bc1bbe1ee3980da5c",
    urls = [
        "https://github.com/openxla/xla/archive/79fd5733f99b3c0948d7202bc1bbe1ee3980da5c.tar.gz",
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

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    requirements = {
        "3.9": "@xla//:requirements_lock_3_9.txt",
        "3.10": "@xla//:requirements_lock_3_10.txt",
        "3.11": "@xla//:requirements_lock_3_11.txt",
        "3.12": "@xla//:requirements_lock_3_12.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

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

load("//xla:workspace.bzl", "workspace")

workspace()
