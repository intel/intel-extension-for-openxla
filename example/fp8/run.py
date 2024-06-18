"""
Copy from https://github.com/keras-team/keras/pull/19488
"""

import argparse
import json

import kagglehub
import keras_nlp

import keras


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gpt2_base_en",
        choices=["gpt2_base_en", "gpt2_medium_en", "gemma_2b_en"],
        help="Which model to demonstrate",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Whether to use float8 technique",
    )
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    args = parser.parse_args()
    return args


def get_optimizer_and_loss():
    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
        epsilon=1e-6,
        global_clipnorm=1.0,  # Gradient clipping.
    )
    # Exclude layernorm and bias terms from weight decay.
    optimizer.exclude_from_weight_decay(var_names=["bias"])
    optimizer.exclude_from_weight_decay(var_names=["gamma"])
    optimizer.exclude_from_weight_decay(var_names=["beta"])

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return optimizer, loss


class GPUMemoryCallback(keras.callbacks.Callback):
    def __init__(self, target_batches, **kwargs):
        super().__init__(**kwargs)
        self.target_batches = target_batches
        self.memory_usage = []

    def _compute_memory_usage(self):
        if keras.backend.backend() == "tensorflow":
            import tensorflow as tf

            memory_stats = tf.config.experimental.get_memory_info("GPU:0")
        else:
            memory_stats = {"peak": 0.0}
        # Convert bytes to GB and store in list.
        peak_usage = round(memory_stats["peak"] / (2**30), 3)
        self.memory_usage.append(peak_usage)

    def on_epoch_begin(self, epoch, logs=None):
        self._compute_memory_usage()

    def on_train_batch_begin(self, batch, logs=None):
        if batch in self.target_batches:
            self._compute_memory_usage()

    def on_epoch_end(self, epoch, logs=None):
        self._compute_memory_usage()


if __name__ == "__main__":
    EPOCHS = 1
    keras.config.disable_traceback_filtering()
    keras.mixed_precision.set_global_policy("mixed_bfloat16")

    args = get_args()
    if args.model == "gemma_2b_en":
        kagglehub.login()

    # Setup dataset
    data = []
    with open("databricks-dolly-15k.jsonl") as file:
        for line in file:
            features = json.loads(line)
            # Filter out examples with context, to keep it simple.
            if features["context"]:
                continue
            # Format the entire example as a single string.
            template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
            data.append(template.format(**features))

    # Only use 320 training examples, to keep it fast.
    train_data = data[:320]
    # Choose the last example to predict.
    eval_data = data[-1:]


    if args.model == "gemma_2b_en":
        preprocessor = keras_nlp.models.GemmaCausalLMPreprocessor.from_preset(
            args.model, sequence_length=128
        )
        model = keras_nlp.models.GemmaCausalLM.from_preset(
            args.model, preprocessor=preprocessor
        )
        model.backbone.token_embedding.trainable = False
    elif "gpt2" in args.model:
        preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
            args.model, sequence_length=128
        )
        model = keras_nlp.models.GPT2CausalLM.from_preset(
            args.model, preprocessor=preprocessor
        )
        model.backbone.token_embedding.trainable = False
        model.backbone.position_embedding.trainable = False
    if args.fp8:
        model.quantize("float8")

    model.summary()
    optimizer, loss = get_optimizer_and_loss()
    model.compile(optimizer=optimizer, loss=loss, weighted_metrics=["accuracy"])
    callbacks = [
        GPUMemoryCallback(target_batches=[5, 10, 25, 50, 100, 150, 200, 300])
    ]
    model.fit(
        train_data, batch_size=args.batch_size, epochs=EPOCHS, callbacks=callbacks
    )
    print("Inference start")
    model.predict(eval_data)

    if keras.backend.backend() == "tensorflow":
        model_memory_usage = callbacks[0].memory_usage
        print(f"GPU Memory Usage (in GB): {max(model_memory_usage)}")

    if args.fp8:
        from keras import layers

        count = 0
        for layer in model._flatten_layers(False, True):
            list_of_sublayers = list(layer._flatten_layers())
            if len(list_of_sublayers) == 1:  # leaves of the model
                if isinstance(layer, (layers.Dense, layers.EinsumDense)):
                    print(
                        layer.outputs_grad_scale.path,
                        layer.outputs_grad_scale.value,
                    )
                    count += 1
                if count > 10:
                    break