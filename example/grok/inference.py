import argparse
import logging
import time
import json
import os
import jax

from model import LanguageModelConfig, TransformerConfig
from runners import InferenceRunner, ModelRunner, sample_from_model

def main(args):
    num_iter = args.num_iter
    num_warmup = args.num_warmup
    input_tokens = args.input_tokens
    max_new_tokens = args.max_new_tokens
    compilcation_cache = args.compilcation_cache
    input_len = int(input_tokens)

    current_path = str(os.path.dirname(__file__))

    if compilcation_cache:
        COMPILATION_CACHE_PATH = current_path +"/compilcation_cache/"
        jax.config.update("jax_compilation_cache_dir", COMPILATION_CACHE_PATH)

    CKPT_PATH = current_path +"/checkpoints/"
    with open(current_path + "/prompt.json") as f:
        content = f.read()
    content_dict = json.loads(content)
    prompt = content_dict[input_tokens]
    
    print("initialize start", flush=True)
    start_time = time.time()
    grok_1_model = LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,
            widening_factor=8,
            key_size=128,
            num_q_heads=48,
            num_kv_heads=8,
            num_layers=64,
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            # MoE.
            num_experts=8,
            num_selected_experts=2,
            # Activation sharding.
            data_axis="data",
            model_axis="model",
        ),
    )
    inference_runner = InferenceRunner(
        pad_sizes=(input_len,),
        runner=ModelRunner(
            model=grok_1_model,
            bs_per_device=0.125,
            checkpoint_path=CKPT_PATH,
        ),
        name="local",
        load=CKPT_PATH,
        tokenizer_path=current_path+"/tokenizer.model",
        local_mesh_config=(1, 8),
        between_hosts_config=(1, 1),
    )
    inference_runner.initialize()
    gen = inference_runner.run()
    end_time = time.time()
    print("initialize time: {:.2f} s".format(end_time - start_time), flush=True)

    step = num_iter + num_warmup
    all_time = 0.0
    for i in range(step):
        print(f"inference start: {i}", flush=True)
        s_time = time.time()
        print(f"Output_{max_new_tokens} for prompt_{input_tokens}:", sample_from_model(gen, prompt, max_len=max_new_tokens, temperature=0.01), flush=True)
        e_time = time.time()
        print("inference time: {:.2f} s".format(e_time - s_time), flush=True)
        if(i >= num_warmup):
            all_time += (e_time - s_time)
    print("averange inference time: {:.2f} s except warmup steps.".format(all_time / float(num_iter)), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iter", default=4, type=int, help="num iter")
    parser.add_argument("--num-warmup", default=1, type=int, help="num warmup")
    parser.add_argument("--input-tokens",default="32",choices=["32", "64", "128", "256", "512", "1024", "2016", "2017", "2048", "4096", "8192"],type=str,help="input tokens length if needed from prompt.json")
    parser.add_argument("--max-new-tokens", default=32, type=int, help="output max new tokens")
    parser.add_argument("--compilcation-cache", default=False, type=bool, help="compilcation cache")
    args = parser.parse_args()
    main(args)
