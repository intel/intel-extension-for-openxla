import os
os.environ["KERAS_BACKEND"] = "jax"
import time
import json
import pathlib
import argparse
import sys
import keras
import keras_nlp
import jax

MODEL_CLASSES = {
  "gemma_2b": "gemma_2b_en",
  "gemma_7b": "gemma_7b_en",
  "gemma_2b_it": "gemma_instruct_2b_en",
  "gemma_7b_it": "gemma_instruct_7b_en",
}
 
parser = argparse.ArgumentParser()
parser.add_argument(
  "--model",
  type=str,
  choices=["gemma_2b", "gemma_7b", "gemma_2b_it", "gemma_7b_it"],
  default="gemma_2b",
  help="the mdoel name",
)
parser.add_argument(
  "--dtype",
  type=str,
  choices=["float32", "bfloat16"],
  default="float32",
  help="bfloat16, float32",
)
parser.add_argument(
  "--input-tokens",
  default="32",
  choices=["32", "64", "128", "256", "512", "1024", "2016", "2017", "2048", "4096", "8192"],
  type=str,
  help="input tokens length if needed from prompt.json",
)
parser.add_argument(
  "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument(
  "--prompt", default=None, type=str, help="input prompt for self-defined if needed"
)
parser.add_argument("--num-beams", default=1, type=int, help="beam width")
parser.add_argument("--num-iter", default=10, type=int, help="num iter")
parser.add_argument("--num-warmup", default=3, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
args = parser.parse_args()

if args.dtype == "bfloat16":
  keras.config.set_floatx("bfloat16")
data_parallel = keras.distribution.DataParallel(devices=keras.distribution.list_devices("sycl"))
keras.distribution.set_distribution(data_parallel)
model = keras_nlp.models.GemmaCausalLM.from_preset(MODEL_CLASSES[args.model])
if args.num_beams > 1:
  from keras_nlp.samplers import BeamSampler
  print("beam")
  model.compile(sampler=BeamSampler(num_beams=args.num_beams))
current_path = os.path.dirname(__file__)
with open(str(current_path) + "/prompt.json") as f:
  prompt_pool = json.load(f)
prompt = prompt_pool[args.input_tokens]

total_time = 0.0
num_iter = args.num_iter
num_warmup = args.num_warmup
num_devices = jax.device_count()
per_device_batch = num_devices * args.batch_size
global_batch = per_device_batch  # will stay 1 unless you implement real multi-device generate
prompt_list = [prompt] * global_batch

total_list = []

for i in range(args.num_warmup):
  output = model.generate(prompt_list, max_length=int(args.max_new_tokens)+int(args.input_tokens))

measured_iters = 0

for i in range(num_iter):
  tic = time.time()

  output = model.generate(
    prompt_list, max_length=int(args.max_new_tokens)+int(args.input_tokens)
  )
  print(output)
  toc = time.time()
  iter_lat = toc - tic
  print(f"Iteration {i}: {iter_lat:.6f} s len of outputs = {len(output)}", flush=True)
  measured_iters += 1
  total_time += iter_lat


avg_iter_latency = total_time / measured_iters
per_sample_latency = avg_iter_latency / global_batch
throughput = global_batch / avg_iter_latency

print("\n---------- Summary (predict forward) ----------", flush=True)
print(f"Average iteration latency: {avg_iter_latency:.6f} s", flush=True)
print(f"Per-sample latency: {per_sample_latency:.6f} s", flush=True)
print(f"Throughput: {throughput:.3f} samples/s", flush=True)