# FP8 training and inference
FP8 becomes increasingly important as models become larger. Here we introduce how to enable fp8 in openxla via keras quantization api.

## Setup Kaggle key to access model
Get your Kaggle key follow [Configure your API key](https://ai.google.dev/gemma/docs/setup#:~:text=T4%20GPU.-,Configure%20your%20API%20key,-To%20use%20Gemma) and then 

```
export KAGGLE_USERNAME=xxxxxxxxx
export KAGGLE_KEY=xxxxxxxx
```

## Package dependency

```bash
pip install keras keras_nlp kagglehub
```

## Dataset

Download [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/main/databricks-dolly-15k.jsonl) dataset.

## Options
```
--model: The model name. Choices are "gpt2_base_en", "gpt2_medium_en", "gemma_2b_en". Default is "gpt2_base_en".
--fp8: Store true. Whether to use float8 technique.
--batch-size: The batch size. Default is 32.
```

## Environment Variables
| **ENV** | **Description** | **PVC Platform** | **ATSM/DG2 Platform** | 
| :---: | :---: | :---: |:---: |
| ZE_AFFINITY_MASK | Run this model on single GPU tile |export ZE_AFFINITY_MASK=0 | export ZE_AFFINITY_MASK=0 |
| XLA_FLAGS | Disable SimplifyFPConversions pass | export XLA_FLAGS="--xla_allow_excess_precision=0" | export XLA_FLAGS="--xla_allow_excess_precision=0" |
| KERAS_BACKEND | Set keras backend | export KERAS_BACKEND=jax | export KERAS_BACKEND=jax |


### Example

```bash
python run.py --model=gpt2_base_en --batch-size=32 --fp8
```

### Expected Output

```
transformer_layer_10/feedforward_output_dense/outputs_grad_scale 2.494614e-09
transformer_layer_10/feedforward_intermediate_dense/outputs_grad_scale 1.4901163e-08
transformer_layer_10/self_attention/attention_output/outputs_grad_scale 1.9374835e-09
transformer_layer_10/self_attention/value/outputs_grad_scale 9.313226e-09
transformer_layer_10/self_attention/key/outputs_grad_scale 4.6898747e-09
```
