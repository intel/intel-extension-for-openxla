#! /bin/bash

pushd ./t5x

git checkout 6699ad54480a0691c491fa2aa28d8b46daf1a377
git apply ../patch/t5.patch

ln -s /usr/local/bin/pip /usr/bin/pip
pip uninstall tensorflow-metadata numba cudf -y

conda install libstdcxx-ng -c conda-forge -y

pip uninstall mdit-py-plugins jupytext -y
pip install t5
pip install -e .

pip install orbax-checkpoint==0.3.2
pip install zstandard==0.21.0
pip install jsonlines==3.1.0

pip uninstall tensorflow tensorflow-cpu tensorflow-text -y
#TensorFlow Text will auto-install the right TensorFlow version as its dependency
pip install tensorflow-text==2.18.1

popd
