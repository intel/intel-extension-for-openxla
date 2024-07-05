# FLAN-T5 Quick Start

## Checkpoint
### Download Checkpoint from URL
#### 1. Chose a pre-trained model
Click the URL attached on checkpoint localtion, and download your picked model manaully. For more T5-like model checkpoints, please got to the official [T5 website](https://t5x.readthedocs.io/en/latest/models.html#public-research-models).
Model                | Gin File Location                                                                                                   | Checkpoint Location
-------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------
Flan-T5 XL    | [t5_1_1_xl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/xl.gin)       | [gs://t5-data/pretrained_models/t5x/flan_t5_xl/checkpoint_1138000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/flan_t5_xl/checkpoint_1138000)
Flan-T5 XXL   | [t5_1_1_xxl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/xxl.gin)     | [gs://t5-data/pretrained_models/t5x/flan_t5_xxl/checkpoint_1114000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/flan_t5_xxl/checkpoint_1114000)

##### How to download the above checkpoint
please follow the officical [guidelines](https://cloud.google.com/storage/docs/gsutil_install#deb) to install gsutil firsty.

```
gsutil -m cp -r "gs://t5-data/pretrained_models/t5x/flan_t5_xl/checkpoint_1138000" .
```

#### 2. Download Vocabulary
T5 Vocabulary: [cc_all.32000.100extra](https://console.cloud.google.com/storage/browser/t5-data/vocabs/cc_all.32000.100extra) 

For more T5-like models, please go to the official [T5 website](https://t5x.readthedocs.io/en/latest/models.html#t5-1-1-checkpoints), and download its corresponding vocabulary manually.

##### How to download the above vocalbulary
```
gsutil -m cp -r "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model" .
```
#### 3.Download Dataset(optional)

We use The Pile for our pretraining experiments. If you would like to as well, run `download_the_pile.py` scipt in `t5x` folder to download it. The download is approximately 1TB.

For benchmarking, you could skip this step because our model script will download a part of dataset automatically.

## Installation

Mark `intel-extension-for-openxla` folder as \<WORKSPACE\>, then
```
cd <WORKSPACE>/example/t5/
git clone https://github.com/google-research/t5x.git
bash install_xpu.sh
pip install --upgrade intel-extension-for-openxla
pip install -r ../../test/requirements.txt
```

## Inference

To fully utilize the hardware capabilities and achieve the best performance, you may consider setting the below ENV variables to enable our customized optimization strategies.

| **ENV** | **Description** | **PVC Platform** | **ATSM/DG2 Platform** | 
| :---: | :---: | :---: |:---: |
| ZE_AFFINITY_MASK | Run this model on single GPU tile |export ZE_AFFINITY_MASK=0 | export ZE_AFFINITY_MASK=0 |
| XETLA_GEMM | Call the [XETLA](https://github.com/intel/xetla) library to run GEMMs, instead of using oneDNN.|export XETLA_GEMM=1 | NA | 
| LLM | Enable our customized optimization strategies for large language models (LLM) |export LLM=1 | export LLM=1 |
| XLA_FLAGS | Customize xla debug options | export XLA_FLAGS="--xla_disable_hlo_passes=dot-merger" | export XLA_FLAGS="--xla_disable_hlo_passes=dot-merger" |

### Command Description
```
bash quick_start.sh [model size] [dataset dir] [model dir] [input_length] [output_length] [device type]
```

#### FLAN-T5-XL Example

#### 32/32
```
bash quick_start.sh xl /username/datasets/ThePile /username/t5x_models 32 32 XPU
```
#### 1024/128
```
bash quick_start.sh xl /username/datasets/ThePile /username/t5x_models 1024 128 XPU
```
#### Performance Output
```
avg time:xxx s,avg throughput:xxx sentences/sencond
```
