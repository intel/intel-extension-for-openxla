Default build cmd is:
```
bazel build //xla/tools/pip_package:build_pip_package
```

This repo pulls public openxla code as its third_party. For development, one often wants to make changes to the XLA repository as well. You can override the pinned xla repo with a local checkout by:
```
bazel build --override_repository=xla=/path/to/xla //xla/tools/pip_package:build_pip_package
```

Then, generate wheel and install it.
```
bazel-bin/xla/tools/pip_package/build_pip_package ./
pip install ./intel_extension_for_openxla-0.1.0-cp39-cp39-linux_x86_64.whl
```

When running jax code, pls `import intel_extension_for_openxla`, otherwise "XPU" device can not be detected.
