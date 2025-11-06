import os
os.environ["KERAS_BACKEND"] = "jax"
import time
import json
from pathlib import Path
import argparse
import sys
import keras
import keras_hub
import jax
import kagglehub

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
  choices=["float32", "float16", "bfloat16"],
  default="float32",
  help="bfloat16, float16, float32",
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

preset = MODEL_CLASSES[args.model]

assets_dir = kagglehub.model_download(f"keras/gemma/keras/{preset}")
spm_path = Path(assets_dir) / "assets" / "tokenizer" / "vocabulary.spm"
if not spm_path.exists():
  raise FileNotFoundError(f"vocabulary.spm not found under: {spm_path}")

# Build tokenizer & preprocessor explicitly (no Keras deserializer involved)
Tokenizer = getattr(keras_hub.models, "GemmaTokenizer")
Preprocessor = getattr(keras_hub.models, "GemmaCausalLMPreprocessor")

with open(spm_path, "rb") as f:
  spm_proto = f.read()

tokenizer = Tokenizer(proto=spm_proto, dtype="int32")
preproc = Preprocessor(tokenizer=tokenizer)

keras.config.set_dtype_policy(args.dtype)
model = keras_hub.models.GemmaCausalLM.from_preset(
    preset, preprocessor=preproc, load_weights=True, dtype=args.dtype
)

if args.num_beams > 1:
  from keras_hub.samplers import BeamSampler
  print("beam")
  model.compile(sampler=BeamSampler(num_beams=args.num_beams))
current_path = os.path.dirname(__file__)
with open(str(current_path) + "/prompt.json") as f:
  prompt_pool = json.load(f)
prompt = prompt_pool[args.input_tokens]

total_time = 0.0
num_iter = args.num_iter
num_warmup = args.num_warmup
prompt = [prompt] * args.batch_size
total_list = []
output = model.generate(prompt, max_length=int(args.max_new_tokens)+int(args.input_tokens))
for i in range(num_iter):
  tic = time.time()
  output = model.generate(
    prompt, max_length=int(args.max_new_tokens)+int(args.input_tokens)
  )
  print(output)
  toc = time.time()
  print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush = True)
  if i >= num_warmup:
    total_time += toc - tic

print("\n", "-" * 10, "Summary:", "-" * 10)
latency = total_time / (num_iter - num_warmup)
print("Inference latency: %.3f sec." % latency)
