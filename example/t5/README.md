# Quick Start

## Checkpoint

### Download checkpoint from URL (external use)
#### 1. Chose a pre-trained model
Click the URL attached on checkpoint localtion, and download your picked model manaully. For more T5-like model checkpoints, please got to the official [T5 website](https://t5x.readthedocs.io/en/latest/models.html#public-research-models).
Model                | Gin File Location                                                                                                   | Checkpoint Location
-------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------
Flan-T5 Small | [t5_1_1/small.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/small.gin) | [gs://t5-data/pretrained_models/t5x/flan_t5_small/checkpoint_1198000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/flan_t5_small/checkpoint_1198000)
Flan-T5 Base  | [t5_1_1/base.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/base.gin)   | [gs://t5-data/pretrained_models/t5x/flan_t5_base/checkpoint_1184000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/flan_t5_base/checkpoint_1184000)
Flan-T5 Large | [t5_1_1_large.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/large.gin) | [gs://t5-data/pretrained_models/t5x/flan_t5_large/checkpoint_1164000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/flan_t5_large/checkpoint_1164000)
Flan-T5 XL    | [t5_1_1_xl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/xl.gin)       | [gs://t5-data/pretrained_models/t5x/flan_t5_xl/checkpoint_1138000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/flan_t5_xl/checkpoint_1138000)
Flan-T5 XXL   | [t5_1_1_xxl.gin](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/xxl.gin)     | [gs://t5-data/pretrained_models/t5x/flan_t5_xxl/checkpoint_1114000](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/flan_t5_xxl/checkpoint_1114000)

##### How to download the above checkpoint
please follow the officical [guidelines](https://cloud.google.com/storage/docs/gsutil_install#deb) to install gsutil firsty.

```
gsutil -m cp -r "gs://t5-data/pretrained_models/t5x/flan_t5_xxl/checkpoint_1114000" .
```

#### 2. Download Vocabulary
T5 Vocabulary: [cc_all.32000.100extra](https://console.cloud.google.com/storage/browser/t5-data/vocabs/cc_all.32000.100extra) 

For more T5-like models, please go to the official [T5 website](https://t5x.readthedocs.io/en/latest/models.html#t5-1-1-checkpoints), and download its corresponding vocabulary manually.

##### How to download the above vocalbulary
```
gsutil -m cp -r "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model" .
```

## Installation

```
git clone https://github.com/google-research/t5x.git
bash install_xpu.sh
```
## Inference

#### Normal mode
```
bash quick_start.sh [model size] [dataset dir] [model dir] [input_length] [output_length] [device type]
```
#### Profile mode
```
bash quick_start.sh [model size] [dataset dir] [model dir] [input_length] [output_length] [device type] [is_profile] [profile out dir]
```

##### example
```
bash quick_start.sh xxl /nfs2/username/datasets/ThePile /nfs2/datasets/t5x_models 1024 1 XPU
```
### Note
If you need generate profile data, you need to build unittrace(https://github.com/intel-sandbox/pti-gpu/tree/master/tools/unitrace) and add it to the PATH:
```
export PATH=$PATH:/path to unit trace bin/
```
##### example
```
bash quick_start.sh xxl /nfs2/username/datasets/ThePile /nfs2/username/t5x_models 1024 1 XPU True
```
