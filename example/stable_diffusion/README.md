# Quick Start for Stable Diffusion Inference

[Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion. More details could be found in the [huggingface](https://huggingface.co/CompVis/stable-diffusion).

## Requirements

### 1. Install intel-extension-for-openxla

please got the [main page](https://github.com/intel/intel-extension-for-openxla/blob/main/README.md#build-and-install), and follow the instructions to build and install intel-extension-for-openxla.

### 2. Install jax
```bash
pip install jax==0.4.20 jaxlib==0.4.20 flax==0.7.0
```
### 3. Install huggingface transformers

```bash
pip install transformers==4.32 diffusers==0.16.1 datasets==2.12.0 msgpack==1.0.7
```
## Run

### 1. Environmental Variables

| **ENV** | **Description** | **PVC Platform** | **ATSM/DG2 Platform** | 
| :---: | :---: | :---: |:---: |
| ZE_AFFINITY_MASK | Run this model on single GPU tile |export ZE_AFFINITY_MASK=0.0 | export ZE_AFFINITY_MASK=0.0 | 
| XLA_FLAGS | Cutomerlize xla debug options | export XLA_FLAGS="--xla_gpu_force_conv_nhwc" | export XLA_FLAGS="--xla_gpu_force_conv_nhwc" |

### 2. Inference Command

```
python jax_stable.py
```

## Expected Output

```
Latency per image is: 1.003s
```