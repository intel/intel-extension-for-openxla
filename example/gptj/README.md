# GPT-J-6B Jax Example

Script jax_gptj.py for [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6b).

## Prerequisites

Mark `intel-extension-for-openxla` folder as \<WORKSPACE\>, then
```bash
cd <WORKSPACE>/example/gptj/
pip install transformers==4.49 datasets==2.20.0
pip install -r ../../test/requirements.txt
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
| *```--num-layer```*| *```28```*| Number of hidden layers. |
| *```--num-warmup```*| *```3```*| Number of warmup iterations. |
| *```--accuracy```*| *```False```*| Run accuracy check. |

## Example

To fully utilize the hardware capabilities and achieve the best performance, you may consider setting the below ENV variables to enable our customized optimization strategies.

| **ENV** | **Description** | **PVC Platform** | **ATSM/DG2 Platform** | 
| :---: | :---: | :---: |:---: |
| ZE_AFFINITY_MASK | Run this model on single GPU tile |export ZE_AFFINITY_MASK=0 | export ZE_AFFINITY_MASK=0 |
| XETLA_GEMM | Call the [XETLA](https://github.com/intel/xetla) library to run GEMMs, instead of using oneDNN.|export XETLA_GEMM=1 | NA |

### Greedy Search

```bash
export ZE_AFFINITY_MASK=0
python jax_gptj.py --greedy
```

### Beam Search = 4

```bash
export ZE_AFFINITY_MASK=0
python jax_gptj.py --input-tokens 1024 --max-new-tokens 128
```

### Performance Output

```
Inference latency: x.xxx sec.
Inference throughput: x.xxx samples/sec.
```

### Accuracy Output

```bash
export ZE_AFFINITY_MASK=0
python jax_gptj.py --input-tokens 1024 --max-new-tokens 128 --accuracy
```

```
Inference latency: x.xxx sec.
Inference throughput: x.xxx samples/sec.

Accuracy = 1.00
```

### Test with less memory

Set option `--num-layer` (default value: `28`) to a small number, to reduce the memory footprint for test.
```bash
export ZE_AFFINITY_MASK=0
python jax_gptj.py --input-tokens 1024 --max-new-tokens 128 --accuracy --num-layer 14
```
