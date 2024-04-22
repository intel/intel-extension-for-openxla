import time
import sys
import argparse

import jax
import numpy as np
from flax.jax_utils import replicate

from diffusers import FlaxStableDiffusionXLPipeline

NUM_DEVICES = jax.device_count()

parser = argparse.ArgumentParser()

parser.add_argument(
  "--dtype",
  type=str,
  choices=["bfloat16", "float16"],
  default="bfloat16",
  help="bfloat16, float16",
)
parser.add_argument("--num-iter", default=1, type=int, help="num iter")
parser.add_argument("--num-inference-steps", default=25, type=int, help="inference steps")

args = parser.parse_args()

dtype = jax.numpy.bfloat16 if args.dtype == "bfloat16" else jax.numpy.float16

pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    revision="refs/pr/95",
    dtype=dtype,
    split_head_dim=True
)

# We cast all parameters to bfloat16 EXCEPT the scheduler which we leave in
# float32 to keep maximal precision
scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(args.dtype), params)
params["scheduler"] = scheduler_state

default_prompt = "a colorful photo of a castle in the middle of a forest with trees and bushes, by Ismail Inceoglu, shadows, high contrast, dynamic shading, hdr, detailed vegetation, digital painting, digital drawing, detailed painting, a detailed digital painting, gothic art, featured on deviantart"
default_neg_prompt = "fog, grainy, purple"
default_seed = 33
default_guidance_scale = 5.0
num_steps = args.num_inference_steps
print("dtype:", args.dtype)

def tokenize_prompt(prompt, neg_prompt):
    prompt_ids = pipeline.prepare_inputs(prompt)
    neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    return prompt_ids, neg_prompt_ids

p_params = replicate(params)

def replicate_all(prompt_ids, neg_prompt_ids, seed):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_neg_prompt_ids, rng

def generate(
    prompt,
    negative_prompt,
    seed=default_seed,
    guidance_scale=default_guidance_scale,
    num_inference_steps=num_steps,
    num_iters=args.num_iter,
):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    
    start = time.time()
    print("Compiling ...")
    images = pipeline(
        prompt_ids,
        p_params,
        rng,
        num_inference_steps=num_inference_steps,
        neg_prompt_ids=neg_prompt_ids,
        guidance_scale=guidance_scale,
        jit=True,
    ).images
    print(f"Compiled in {time.time() - start}")

    start = time.time()
    for i in range(num_iters):
        cur = time.time()
        images = pipeline(
            prompt_ids,
            p_params,
            rng,
            num_inference_steps=num_inference_steps,
            neg_prompt_ids=neg_prompt_ids,
            guidance_scale=guidance_scale,
            jit=True,
        ).images
        print("Latency of iter {}: {:.3f}s".format(i, time.time() - cur), file=sys.stderr)
    
    end = time.time()
    return (end - start) / num_iters, images

latency, images = generate(default_prompt, default_neg_prompt)
print("Average Latency per image is: {:.3f}s".format(latency), file=sys.stderr)
images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
images = pipeline.numpy_to_pil(np.array(images))

for i, image in enumerate(images):
    image.save(f"castle_{i}.png")
