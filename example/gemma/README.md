# Gemma Jax Example
[Gemma](https://www.kaggle.com/models/google/gemma) is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights, pre-trained variants, and instruction-tuned variants. Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as a laptop, desktop or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.

## Setup Kaggle key to access model
Get your Kaggle key follow [Configure your API key](https://ai.google.dev/gemma/docs/setup#:~:text=T4%20GPU.-,Configure%20your%20API%20key,-To%20use%20Gemma) and then 

```
export KAGGLE_USERNAME=xxxxxxxxx
export KAGGLE_KEY=xxxxxxxx
```

## Inference

### Prerequisites

```bash
pip install jax==0.4.25 jaxlib==0.4.25 keras-nlp==0.10.0 keras==3.3.2
```

### Options
```
--model: The model name. Choices are "gemma_2b", "gemma_7b", "gemma_2b_it", "gemma_7b_it". Default is "gemma_2b".
--dtype: The data type. Choices are "float32", "bfloat16". Default is "float32".
--input-tokens: The length of input tokens. Choices are "32", "64", "128", "256", "512", "1024", "2016", "2017", "2048", "4096", "8192". Default is "32".
--max-new-tokens: The maximum number of new tokens to generate. Default is 32.
--prompt: The input prompt for self-defined if needed.
--num-beams: The beam width for beam search. Default is 1.
--num-iter: The number of iterations to run. Default is 10.
--num-warmup: The number of warmup iterations. Default is 3.
--batch-size: The batch size. Default is 1.
```

### Environment Variables
| **ENV** | **Description** | **PVC Platform** | **ATSM/DG2 Platform** | 
| :---: | :---: | :---: |:---: |
| ZE_AFFINITY_MASK | Run this model on single GPU tile |export ZE_AFFINITY_MASK=0 | export ZE_AFFINITY_MASK=0 |


### Example

```bash
numactl -N 0 -m 0 python inference.py --model gemma_7b --dtype bfloat16 --input-tokens 32 --max-new-tokens 32
numactl -N 0 -m 0 python inference.py --model gemma_7b --dtype bfloat16 --input-tokens 1024 --max-new-tokens 128
numactl -N 0 -m 0 python inference.py --model gemma_2b --dtype bfloat16 --input-tokens 32 --max-new-tokens 32
numactl -N 0 -m 0 python inference.py --model gemma_2b --dtype bfloat16 --input-tokens 1024 --max-new-tokens 128
numactl -N 0 -m 0 python inference.py --model gemma_7b --dtype bfloat16 --input-tokens 32 --max-new-tokens 32 --num-beams 4
numactl -N 0 -m 0 python inference.py --model gemma_7b --dtype bfloat16 --input-tokens 1024 --max-new-tokens 128 --num-beams 4
numactl -N 0 -m 0 python inference.py --model gemma_2b --dtype bfloat16 --input-tokens 32 --max-new-tokens 32 --num-beams 4
numactl -N 0 -m 0 python inference.py --model gemma_2b --dtype bfloat16 --input-tokens 1024 --max-new-tokens 128 --num-beams 4
```

### Output
```
Inference latency: xxx.xxx sec.
```

## Finetune

### Prerequisites
```
pip install -q tensorflow-cpu
pip install -q -U keras-nlp tensorflow-hub
pip install -q -U keras>=3
pip install -U tensorflow-text
```

### Options

```
--model: The model name. Choices are "gemma_2b", "gemma_7b", "gemma_2b_it", "gemma_7b_it". Default is "gemma_7b".
--dtype: The data type. Choices are "float32", "bfloat16". Default is "bfloat16".
--seqlen: Length of training sequences. Default is 256.
--batchsize_per_gpu: Batch size per gpu. Default is 4.
--steps: Steps of finetuning. Default is 300.
--warmup_steps: Steps of warm up. Default is 100.
--num_gpus: Number of gpus to use. Default is 2.
--data_parallel: Number of data parallel. Default is 2.
--model_parallel: Number of model tensor parallel. Default is 1.
--use_lora: Finetune with LoRA.
--lora_ranks: Ranks of LoRA. Default is 4.
```

**Note**: Please make sure `num_gpus` equals to `data_parallel x model_parallel`. We recommand to use pure data parallel (which means `num_gpus` equals to `data_parallel`, and set `model_parallel` as 1) for better performance if your GPUs has enough memory.

### Examples

#### Parameters of Single GPU finetune.
```
python finetune.py \
--model gemma_7b \
--dtype bfloat16 \
--seqlen 256 \
--batchsize_per_gpu 4 \
--steps 300 \
--warmup_steps 100 \
--num_gpus 1 \
--data_parallel 1 \
--model_parallel 1 \
--use_lora \
--lora_rank 4 
```

#### Parameters of Multiple GPUs data parallel distributed finetune

```
python finetune.py \
--model gemma_7b \
--dtype bfloat16 \
--seqlen 256 \
--batchsize_per_gpu 4 \
--steps 300 \
--warmup_steps 100 \
--num_gpus 8 \
--data_parallel 8 \
--model_parallel 1 \
--use_lora \
--lora_rank 4 
```

**Note**: You could edit `num_gpus` and `data_parallel` to the number of GPUs you want to use.

### Output
```
[INFO] Average Lantency of each steps is : xxx.xxx ms.
```
