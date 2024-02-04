# Quick Start for ResNet50 Training

Trains ResNet50 model (He et al., 2016) for the ImageNet classification task (Russakovsky et al., 2015) by [FLAX RN50 example](https://github.com/google/flax/tree/main/examples/imagenet)

## Requirements

### 1. Install intel-extension-for-openxla

Please got the [main page](https://github.com/intel/intel-extension-for-openxla/blob/main/README.md#build-and-install), and follow the instructions to build and install intel-extension-for-openxla.

### 2. Install jax and flax
```bash
pip install jax==0.4.20 jaxlib==0.4.20
```
### 3. Install dependency
```bash
git clone --branch=main https://github.com/google/flax
cd flax
git checkout ba9e24a7b697e6407465cb4b05a24a4cea152248
cd examples/imagenet
pip install -r requirements.txt
```
### 4. Download dataset

Please follow [Preparing the dataset](https://github.com/google/flax/tree/main/examples/imagenet#preparing-the-dataset) to get imagenet dataset.

## Run

### Set environment
```bash
export PYTHONPATH=${path_to_flax}
```

### Running command
```bash
python main.py --workdir=./imagenet --config=configs/default.py
```
`config.batch_size` is global batchsize for all devices you selected.

### Select devices
All GPU devices in same node will be used by default. If you only want some of devices, please use environmental variable `ZE_AFFINITY_MASK` to select.

| **ENV** | **Description** | **PVC Platform** |
| :---: | :---: | :---: |
| ZE_AFFINITY_MASK | Run this model on single GPU device |export ZE_AFFINITY_MASK as your selected device list, like 0,1,2,3|
