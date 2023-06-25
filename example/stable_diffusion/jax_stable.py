import jax
import sys
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline, FlaxDPMSolverMultistepScheduler
import time
from PIL import Image

scheduler, scheduler_state = FlaxDPMSolverMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, revision="bf16", dtype=jax.numpy.bfloat16)
params["scheduler"] = scheduler_state

prompt = "a photo of an astronaut riding a horse on mars"

prng_seed = jax.random.PRNGKey(0)

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

def elapsed_time(nb_pass=10, num_inference_steps=20):
    # warmup
    images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
    start = time.time()
    for _ in range(nb_pass):
        images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
    end = time.time()
    return (end - start) / nb_pass, images

latency, images = elapsed_time(nb_pass=10, num_inference_steps=20)
print("Latency per image is: {:.3f}s".format(latency), file=sys.stderr)

