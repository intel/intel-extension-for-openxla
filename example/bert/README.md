# Quick Start for fine-tunes BERT on SQuAD
Fine-tunes BERT model on SQuAD task by [Question Answering examples](https://github.com/huggingface/transformers/tree/v4.32.0/examples/flax/question-answering#question-answering-examples).
This expample is referred from [HuggingFace Transformers](https://github.com/huggingface/transformers). See [Backup](#Backup) for modification details.


**IMPORTANT: This example is temporarily unavailable under JAX v0.4.20 with below error due to public issue (https://github.com/huggingface/transformers/issues/27644):**
```
AttributeError: 'ArrayImpl' object has no attribute 'split'
```
**Will reenable it once it's fixed in community.**


## Requirements

### 1. Install intel-extension-for-openxla
Please got the [main page](https://github.com/intel/intel-extension-for-openxla/blob/main/README.md#build-and-install), and follow the instructions to build and install `intel-extension-for-openxla`.

### 2. Install dependency
Mark `intel-extension-for-openxla` folder as \<WORKSPACE\>, then
```bash
pip install jax==0.4.20 jaxlib==0.4.20 flax==0.7.0
cd <WORKSPACE>/example/bert
pip install -r requirements.txt
```

### 3. Download pre-trained model
```bash
mkdir models
cd models
wget https://huggingface.co/bert-large-uncased/blob/main/flax_model.msgpack
wget https://huggingface.co/bert-large-uncased/blob/main/config.json
wget https://huggingface.co/bert-large-uncased/resolve/main/tokenizer.json
wget https://huggingface.co/bert-large-uncased/resolve/main/tokenizer_config.json
cd -
```

## Run

### Running command
```bash
python run_qa.py \
 --model_name_or_path <WORKSPACE>/example/bert/models \
 --dataset_name squad \
 --do_train \
 --per_device_train_batch_size 8 \
 --learning_rate 3e-5 \
 --num_train_epochs 1 \
 --warmup_steps 50 \
 --max_seq_length 384 \
 --doc_stride 128 \
 --output_dir /tmp/wwm_uncased_finetuned_squad/ \
 --max_train_samples 800
```
* `per_device_train_batch_size` is local batchsize for each device you selected.
* `max_train_samples` is total training samples to be run. Note `total_steps = max_train_samples / (per_device_train_batch_size * devices)`

### Select devices
All GPU devices in same node will be used by default. If you only want some of devices, please use environmental variable `ZE_AFFINITY_MASK` to select.

| **ENV** | **Description** | **PVC Platform** |
| :---: | :---: | :---: |
| ZE_AFFINITY_MASK | Run this model on single GPU device |export ZE_AFFINITY_MASK as your selected device list, like 0,1,2,3|

### Check performance
Throughput will be printed after the model is completed as below:
```
Performance... xxx iter/s
```

### Backup
```patch
diff --git a/examples/flax/question-answering/run_qa.py b/examples/flax/question-answering/run_qa.py
index a2839539e..a530d8560 100644
--- a/examples/flax/question-answering/run_qa.py
+++ b/examples/flax/question-answering/run_qa.py
@@ -846,7 +846,8 @@ def main():

     # region Training steps and logging init
     train_dataset = processed_raw_datasets["train"]
-    eval_dataset = processed_raw_datasets["validation"]
+    if training_args.do_eval:
+        eval_dataset = processed_raw_datasets["validation"]

     # Log a few random samples from the training set:
     for index in random.sample(range(len(train_dataset)), 3):
@@ -957,11 +958,12 @@ def main():
     state = replicate(state)

     train_time = 0
+    total_train_time = 0
     step_per_epoch = len(train_dataset) // train_batch_size
     total_steps = step_per_epoch * num_epochs
     epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
     for epoch in epochs:
-        train_start = time.time()
+        last_time = train_start = time.time()
         train_metrics = []

         # Create sampling rng
@@ -982,6 +984,13 @@ def main():

             cur_step = epoch * step_per_epoch + step

+            # Print performance result
+            cur_time = time.time()
+            if cur_step > training_args.warmup_steps:
+                total_train_time += (cur_time - last_time)
+            last_time = cur_time
+
+
             if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                 # Save metrics
                 train_metric = unreplicate(train_metric)
@@ -1048,6 +1057,9 @@ def main():
                     if training_args.push_to_hub:
                         repo.push_to_hub(commit_message=f"Saving weights and logs of step {cur_step}", blocking=False)
         epochs.desc = f"Epoch ... {epoch + 1}/{num_epochs}"
+        throughput = format((total_steps - training_args.warmup_steps) / total_train_time, '.4f')
+        epochs.write(f"Performance... {throughput} iter/s")
+
     # endregion

     # Eval after training
```
