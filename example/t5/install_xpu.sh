#! /bin/bash

pushd ./t5x

git checkout 6699ad54480a0691c491fa2aa28d8b46daf1a377
git apply ../patch/t5.patch

ln -s /usr/local/bin/pip /usr/bin/pip
pip uninstall tensorflow-metadata numba cudf -y
pip uninstall tensorflow -y
pip install tensorflow==2.18.0

conda install libstdcxx-ng==12.2.0 -c conda-forge -y

pip uninstall mdit-py-plugins jupytext -y
pip install t5
pip install -e .

pip install orbax-checkpoint==0.3.2
pip install zstandard==0.21.0
pip install jsonlines==3.1.0

popd
