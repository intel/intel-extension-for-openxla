# GPT-J-6B Jax Example

Script jax_gptj.py for [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6b).

## Prerequisites

```bash
pip install jax==0.4.13 jaxlib==0.4.13 flax==0.7.0 transformers==4.27.4 diffusers==0.16.1 datasets==2.12.0
```

## Options

| Option | Default Value | Description|
| :-- | :--: | :--: |
| *```--dtype```*| *```float16```*| Data type, support *```float16```*, *```bfloat16```*, and *```float32```*. |
| *```--batch-size```*| *```1```*| Batch size |
| *```--prompt```*| *```None```*| Customized prompt, not supported  when *```--accuracy-only```* is on. |
| *```--input-tokens```*| *```32```*| Input tokens. |
| *```--max-new-tokens```*| *```32```*| Output max new tokens. |
| *```--greedy```*| *```False```*| Enable greedy search or beam search. |
| *```--num-iter```*| *```10```*| Number of iterations. |
| *```--num-warmup```*| *```3```*| Number of warmup iterations. |
| *```--accuracy-only```*| *```False```*| Run for performance or accuracy only. |

## Accuracy Example

```bash
export ZE_AFFINITY_MASK=0.0
numactl -N 0 -m 0 python jax_gptj.py --accuracy-only --dtype "float16"
```

## Performance Example

### Greedy Search

```bash
export ZE_AFFINITY_MASK=0.0
numactl -N 0 -m 0 python jax_gptj.py --greedy
```

### Beam Search = 4

```bash
export ZE_AFFINITY_MASK=0.0
numactl -N 0 -m 0 python jax_gptj.py --input-tokens 1024 --max-new-tokens 128 --num-iter 100 --num-warmup 10
```
