import argparse
import os
import time
os.environ["KERAS_BACKEND"] = "jax"
import keras
import keras_nlp

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
    default="bfloat16",
    help="bfloat16, float32",
)

parser.add_argument(
    "--seqlen", 
    default=256, 
    type=int, 
    help="Seqlen"
)

parser.add_argument(
    "--batchsize_per_gpu", 
    default=1, 
    type=int, 
    help="Batch size per gpu"
)

parser.add_argument(
    "--steps", 
    default=300, 
    type=int, 
    help="Steps")

parser.add_argument(
    "--warmup_steps", 
    default=100, 
    type=int, 
    help="Warmup steps")

parser.add_argument(
    "--num_gpus",
    type=int,
    default=1,
    help="Number of gpus to use",
)

parser.add_argument(
    "--data_parallel",
    type=int,
    default=0,
    help="Number of the model to be replicated. Cowork with --model_parallel. Please make sure number of GPUs on this host is data_parallel x model_parallel!",
)

parser.add_argument(
    "--model_parallel",
    type=int,
    default=0,
    help="Number of the model to be shared. Cowork with --tensor_parallel. Please make sure number of GPUs on this host is data_parallel x model_parallel!",
)

parser.add_argument(
    "--use_lora",
    action="store_true",
    help="Perform finetuning using Low Rank Adaptation (LoRA).",
)

parser.add_argument(
  "--lora_rank",
  type=int,
  default=4,
  help="Rank of LoRA",
)

args = parser.parse_args()

class PerformanceCallback(keras.callbacks.Callback):
    def __init__(self, warmup_steps):
        super().__init__()
        self.warmup_steps = warmup_steps

    def on_batch_begin(self, batch, logs=None):
        self.total_steps += 1
        if self.total_steps > self.warmup_steps:
            self.begin_time_ns = time.time_ns()

    def on_batch_end(self, batch, logs=None):
        if self.total_steps > self.warmup_steps:
            self.total_time += time.time_ns() - self.begin_time_ns

    def on_train_begin(self, logs=None):
        self.total_steps = 0
        self.total_time = 0

    def on_train_end(self, logs=None):
        print("[INFO] Average Lantency of each steps is : {} ms.".format(str(self.total_time/(self.total_steps-self.warmup_steps)/1000000)))

print("[INFO] Using devices:", keras.distribution.list_devices("xpu")[:args.num_gpus])

if args.dtype == "bfloat16":
    print("[INFO] Using bfloat16 datatype.")
    keras.config.set_floatx("bfloat16")

if args.data_parallel == 0 and args.model_parallel == 0:
    args.data_parallel = args.num_gpus
    args.model_parallel = 1
    if args.num_gpus > 1:
        print("[INFO] Using all {} xpu devices for data parallel by default.".format(str(args.num_gpus)))

if args.data_parallel * args.model_parallel != args.num_gpus:
    raise RuntimeError("Please make sure num_gpus is equal to data_parallel x model_parallel!")

if args.num_gpus > 1:
    print("[INFO] Using data_parallel = {} and model_parallel = {}.".format(str(args.data_parallel), str(args.model_parallel)))
    if args.model_parallel == 1:
        data_parallel = keras.distribution.DataParallel(devices=keras.distribution.list_devices("xpu")[:args.num_gpus])
        keras.distribution.set_distribution(data_parallel)
    else:
        device_mesh = keras.distribution.DeviceMesh(
            (args.data_parallel, args.model_parallel),
            ["batch", "model"],
            devices=keras.distribution.list_devices("xpu")[:args.num_gpus])

        model_dim = "model"
        layout_map = keras.distribution.LayoutMap(device_mesh)
        layout_map["token_embedding/embeddings"] = (None, model_dim)
        layout_map["decoder_block.*attention.*(query|key|value).*kernel"] = (
            None, model_dim, None)
        layout_map["decoder_block.*attention_output.*kernel"] = (
            None, None, model_dim)
        layout_map["decoder_block.*ffw_gating.*kernel"] = (model_dim, None)
        layout_map["decoder_block.*ffw_linear.*kernel"] = (None, model_dim)

        model_parallel = keras.distribution.ModelParallel(
            device_mesh, layout_map, batch_dim_name="batch")
        keras.distribution.set_distribution(model_parallel)

# Load model
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(MODEL_CLASSES[args.model])

# Fine-tune on the IMDb movie reviews dataset.
import tensorflow_datasets as tfds
print("[INFO] Global batch size is {}.".format(str(args.batchsize_per_gpu * args.data_parallel)))
imdb_train = tfds.load(
    "imdb_reviews",
    split="train",
    as_supervised=True,
    batch_size=args.batchsize_per_gpu * args.data_parallel,
)
imdb_train = imdb_train.map(lambda x, y: x) # Drop labels.
imdb_train = imdb_train.take(args.steps)

if args.use_lora:
    gemma_lm.backbone.enable_lora(rank=args.lora_rank)  
gemma_lm.preprocessor.sequence_length = args.seqlen

optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5 * args.data_parallel,
    weight_decay=0.01,
)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.summary()
gemma_lm.fit(imdb_train, epochs=1, callbacks=[PerformanceCallback(args.warmup_steps)])
