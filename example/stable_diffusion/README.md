```bash
git clone https://github.com/intel-innersource/frameworks.ai.intel-extension-for-openxla.intel-extension-for-openxla
git checkout yang/mha

pip install jax==0.4.7 jaxlib==0.4.7 flax==0.6.6 transformers==4.27.4 diffusers==0.16.1 datasets==2.12.0
./configure
bazel build --verbose_failures //xla:libitex_xla_extension.so

export LD_LIBRARY_PATH="/python-path/lib/python3.9/site-packages/jaxlib/:$LD_LIBRARY_PATH"
export PJRT_NAMES_AND_LIBRARY_PATHS="xpu:/xla-path/bazel-bin/xla/libitex_xla_extension.so"

export ZE_AFFINITY_MASK=0.0
numactl -N 0 -m 0 python jax_stable.py
```
