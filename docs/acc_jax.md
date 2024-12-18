# Accelerated JAX on Intel GPU

## Intel® Extension for OpenXLA* plug-in
Intel® Extension for OpenXLA* includes PJRT plugin implementation, which seamlessly runs JAX models on Intel GPU. The PJRT API simplified the integration, which allowed the Intel GPU plugin to be developed separately and quickly integrated into JAX. Refer to [OpenXLA PJRT Plugin RFC](https://github.com/openxla/community/blob/main/rfcs/20230123-pjrt-plugin.md) for more details.

## Requirements
Please check [README#requirements](../README.md#2-requirements) for the requirements of hardware and software.

## Install
The following table tracks intel-extension-for-openxla versions and compatible versions of `jax` and `jaxlib`. The compatibility between `jax` and `jaxlib` is maintained through JAX. This version restriction will be relaxed over time as the plugin API matures.
|**intel-extension-for-openxla**|**jaxlib**|**jax**|
|:-:|:-:|:-:|
| 0.5.0 | 0.4.30 | >= 0.4.30, <= 0.4.31|
| 0.4.0 | 0.4.26 | >= 0.4.26, <= 0.4.27|
| 0.3.0 | 0.4.24 | >= 0.4.24, <= 0.4.27|
| 0.2.1 | 0.4.20 | >= 0.4.20, <= 0.4.26|
| 0.2.0 | 0.4.20 | >= 0.4.20, <= 0.4.26|
| 0.1.0 | 0.4.13 | >= 0.4.13, <= 0.4.14|

[conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) is recommanded as the virtual running environment.
```
conda create -n jax-ioex python=3.10
conda activate jax-ioex
pip install -U pip
pip install intel-extension-for-openxla

# Install jax, jaxlib and flax dependency
pip install -r https://raw.githubusercontent.com/intel/intel-extension-for-openxla/main/test/requirements.txt
```
Please refer to [requirements.txt](../test/requirements.txt) for the version dependency of `jax`, `jaxlib` and `flax`.

## Verify
```
python -c "import jax; print(jax.devices())"
```
Reference result:
```
[xpu(id=0), xpu(id=1)]
```

## Example - Run Stable Diffusion Inference

### Install miniforge
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

### Setup environment
Please follow [Install](#install) to prepare the basic environment first.
```
pip install transformers==4.47 diffusers==0.31.0 datasets==4.9.7 msgpack==1.1.0
```
Source OneAPI env
```
source /opt/intel/oneapi/compiler/2025.0/env/vars.sh
source /opt/intel/oneapi/mkl/2025.0/env/vars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/umf/latest/lib
```
**NOTE**: The path of OneAPI env script is based on the OneAPI installed path.

### Run Demo (Stable Diffusion Inference)
Go to [example/stable_diffusion](../example/stable_diffusion/README.md) for detail about this demo.

| **Command** | **Model** | **Output Image Resolution** | 
| :--- | :---: | :---: |
| ```python jax_stable.py``` | [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | 512x512 |
| ```python jax_stable.py -m stabilityai/stable-diffusion-2``` | [stabilityai/stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) | 768x768 |
| ```python jax_stable.py -m stabilityai/stable-diffusion-2-1``` | [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) | 768x768 |

### Expected result:
```
Average Latency per image is: x.xxx s
Average Throughput per second is: x.xxx steps
```

## Support
To submit questions, feature requests, and bug reports about the intel-extension-for-openxla plugin, visit the [GitHub intel-extension-for-openxla issues](https://github.com/intel/intel-extension-for-openxla/issues) page. You can also view [GitHub JAX Issues](https://github.com/google/jax/issues) with the label "Intel GPU plugin".

## FAQ

1. If there is an error 'No visible XPU devices', print `jax.local_devices()` to check which device is running. Set `export OCL_ICD_ENABLE_TRACE=1` to check if there are driver error messages. The following code opens more debug log for JAX app.

    ```python
    import logging
    logging.basicConfig(level = logging.DEBUG)
    ```

2. If there is an error 'version GLIBCXX_3.4.30' not found, upgrade libstdc++ to the latest, for example for conda

    ```bash
    conda install libstdcxx-ng==12.2.0 -c conda-forge
    ```
