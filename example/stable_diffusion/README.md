# Quick Start for Stable Diffusion Inference

[Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model. More details could be found in [Stable Diffusion v1](https://github.com/CompVis/stable-diffusion) and [Stable Diffusion v2](https://github.com/Stability-AI/stablediffusion).

## Requirements

### 1. Install intel-extension-for-openxla

please got the [main page](https://github.com/intel/intel-extension-for-openxla/blob/main/README.md#build-and-install), and follow the instructions to build and install intel-extension-for-openxla.

### 2. Install packages

Mark `intel-extension-for-openxla` folder as \<WORKSPACE\>, then
```bash
cd <WORKSPACE>/example/stable_diffusion/
pip install transformers==4.48 diffusers==0.26.3 datasets==2.20.0 msgpack==1.0.7
pip install -r ../../test/requirements.txt
```

## Run

### 1. Environmental Variables

| **ENV** | **Description** | **PVC Platform** | **ATSM/DG2 Platform** | 
| :---: | :---: | :---: |:---: |
| ZE_AFFINITY_MASK | Run this model on single GPU tile |export ZE_AFFINITY_MASK=0 | export ZE_AFFINITY_MASK=0 |
| XLA_FLAGS | Customize xla debug options | export XLA_FLAGS="--xla_gpu_force_conv_nhwc" | export XLA_FLAGS="--xla_gpu_force_conv_nhwc" |

### 2. Inference Command

| **Model** | **Output Image Resolution** | **Command** | 
| :---: | :---: | :---: |
| [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | 512x512 | ```python jax_stable.py``` |
| [stabilityai/stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) | 768x768 | ```python jax_stable.py -m stabilityai/stable-diffusion-2``` |
| [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) | 768x768 | ```python jax_stable.py -m stabilityai/stable-diffusion-2-1``` |

Add option `--accuracy` to check whether the demo result is expected. Output range is `0`~`1`, higher is better:
```shell
python jax_stable.py -m stabilityai/stable-diffusion-2-1 --accuracy
```

## Expected Output

### Performance
```
Average Latency per image is: x.xxx s
Average Throughput per second is: x.xxx steps
```

### Accuracy
```
RMSE accuracy is: 0.976
```
