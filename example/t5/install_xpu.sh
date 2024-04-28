#! /bin/bash

pushd ./t5x

git checkout 6699ad54480a0691c491fa2aa28d8b46daf1a377
git apply ../patch/not_exit_before_max_step.patch
git apply ../patch/version_time_dlpath.patch
git apply ../patch/adjust_flax.patch
git apply ../patch/adjust_jax.patch

ln -s /usr/local/bin/pip /usr/bin/pip
pip uninstall tensorflow-metadata numba cudf -y
pip uninstall tensorflow -y
pip install tensorflow==2.12.0

conda install libstdcxx-ng==12.2.0 -c conda-forge -y
pip install jax==0.4.25 jaxlib==0.4.25

pip uninstall mdit-py-plugins jupytext -y
pip install t5
pip install -e .

pip install flax==0.8.2
pip install orbax-checkpoint==0.3.2
pip install zstandard==0.21.0
pip install jsonlines==3.1.0

popd
