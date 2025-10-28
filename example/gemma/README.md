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

Mark `intel-extension-for-openxla` folder as \<WORKSPACE\>, then
```bash
cd <WORKSPACE>/example/gemma/
pip install keras==3.3.2
git clone https://github.com/keras-team/keras-nlp.git
cd keras-nlp
git checkout v0.10.0
git apply ../keras_nlp.patch
python setup.py install
cd ..
pip install -r ../../test/requirements.txt
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
python inference.py --model gemma_7b --dtype bfloat16 --input-tokens 32 --max-new-tokens 32
python inference.py --model gemma_7b --dtype bfloat16 --input-tokens 1024 --max-new-tokens 128
python inference.py --model gemma_2b --dtype bfloat16 --input-tokens 32 --max-new-tokens 32
python inference.py --model gemma_2b --dtype bfloat16 --input-tokens 1024 --max-new-tokens 128
python inference.py --model gemma_7b --dtype bfloat16 --input-tokens 32 --max-new-tokens 32 --num-beams 4
python inference.py --model gemma_7b --dtype bfloat16 --input-tokens 1024 --max-new-tokens 128 --num-beams 4
python inference.py --model gemma_2b --dtype bfloat16 --input-tokens 32 --max-new-tokens 32 --num-beams 4
python inference.py --model gemma_2b --dtype bfloat16 --input-tokens 1024 --max-new-tokens 128 --num-beams 4
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

#### Parameters for Single GPU finetune.
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

#### Parameters for Multiple GPUs data parallel distributed finetune

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

## Accuracy
Based on lm-eval-harness.
### Script Parameters

| **Parameter** | **Default Value** |
| :---: | :--- |
| **model** | gemma |
| **model_name** | **gemma_2b_en**,  gemma_7b_en| 
| **dtype** | **bfloat16**, float16, float32 |
| **num_beams** | **1** |
```
git clone https://github.com/EleutherAI/lm-evaluation-harness.git lm_eval
cd lm_eval
git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463
git apply ../gemma.patch
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall
export KERAS_BACKEND=jax
python main.py \
  --model gemma \
  --model_args model_name=gemma_7b_en,dtype=bfloat16,num_beams=4 \
  --tasks openbookqa \
  --no_cache
```
### Output
```
gemma (model_name=gemma_7b_en,dtype=bfloat16,num_beams=4), limit: None, provide_description: False, num_fewshot: 0, batch_size: None
|   Task   |Version| Metric |Value|   |Stderr|
|----------|------:|--------|----:|---|-----:|
|openbookqa|      0|acc     |0.326|±  |0.0210|
|          |       |acc_norm|0.454|±  |0.0223|
```
