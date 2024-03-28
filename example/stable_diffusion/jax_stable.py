# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import jax
import sys
import requests

import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline, FlaxDPMSolverMultistepScheduler, FlaxStableDiffusionImg2ImgPipeline
import time
from PIL import Image
from io import BytesIO
import argparse

# args
parser = argparse.ArgumentParser("Stable diffusion generation script", add_help=False)
parser.add_argument("-m", "--model-id", default="CompVis/stable-diffusion-v1-4", type=str, 
    choices=["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2", "stabilityai/stable-diffusion-2-1"])
parser.add_argument("--num-inference-steps", default=50, type=int, help="inference steps")
parser.add_argument("--num-iter", default=10, type=int, help="num iter")
parser.add_argument("--profile", action="store_true")
parser.add_argument('--pipeline_mode', choices=["img2img", "text2img"],
                    default="text2img", type=str, help='evaluation method')
args = parser.parse_args()
print(args, file=sys.stderr)

model_id = args.model_id
scheduler, scheduler_state = FlaxDPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
if args.pipeline_mode == "img2img":
    pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, revision="bf16", safety_checker=None, feature_extractor=None, dtype=jax.numpy.bfloat16)
else:
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="bf16", safety_checker=None, feature_extractor=None, dtype=jax.numpy.bfloat16)
params["scheduler"] = scheduler_state

prompt = "a photo of an astronaut riding a horse on mars"

if args.pipeline_mode == "img2img":
    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((768, 768))
    print(init_image)
    prompt = "A fantasy landscape, trending on artstation"

prng_seed = jax.random.PRNGKey(0)

num_samples = jax.device_count()
prompt = num_samples * [prompt]
init_image = num_samples * [init_image]
prompt_ids, processed_image = pipeline.prepare_inputs(prompt, init_image)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)
processed_image = shard(processed_image)

def elapsed_time(num_iter=10, num_inference_steps=20):
    # warmup
    if args.pipeline_mode == "img2img":
        images = pipeline(prompt_ids, processed_image, params, prng_seed, num_inference_steps=num_inference_steps, jit=True).images
    else:
        images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
    start = time.time()
    if args.profile:
        jax.profiler.start_trace("./trace")
    for i in range(num_iter):
        cur = time.time()
        if args.pipeline_mode == "img2img":
            images = pipeline(prompt_ids, processed_image, params, prng_seed, num_inference_steps=num_inference_steps, jit=True).images
        else:
            images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
        print("Latency of iter {}: {:.3f}s".format(i, time.time() - cur), file=sys.stderr)
    if args.profile:
        jax.profiler.stop_trace()
    end = time.time()
    return (end - start) / num_iter, images


num_inference_steps = args.num_inference_steps
num_iter = args.num_iter
latency, images = elapsed_time(num_iter, num_inference_steps)
print("Average Latency per image is: {:.3f}s".format(latency), file=sys.stderr)
images = images.reshape((images.shape[0],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)
images[0].save("img.png")
