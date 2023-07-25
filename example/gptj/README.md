```bash
pip install jax==0.4.13 jaxlib==0.4.13 flax==0.7.0 transformers==4.27.4 diffusers==0.16.1 datasets==2.12.0

export ZE_AFFINITY_MASK=0.0
numactl -N 0 -m 0 python jax_gptj.py
```
