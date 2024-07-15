# Quick Start for Grok Inference

Loading and running the Grok-1 open-weights model by [Grok-1](https://github.com/xai-org/grok-1)

The Grok-1 model running needs at least 8-tile GPU device.

## 1. Install intel-extension-for-openxla

Please got the [main page](https://github.com/intel/intel-extension-for-openxla/blob/main/README.md#build-and-install), and follow the instructions to build and install intel-extension-for-openxla.

## 2. Install dependency

Mark `intel-extension-for-openxla` folder as \<WORKSPACE\>, then
```bash
cd <WORKSPACE>/example/grok/
git clone https://github.com/xai-org/grok-1.git
pip install -r grok-1/requirements.txt
pip install -r ../../test/requirements.txt
```

## 3. Download the weights

Please follow [Downloading the weights](https://github.com/xai-org/grok-1#downloading-the-weights) to get the weights.

Make sure to download the checkpoint and place the ckpt-0 directory in grok-1/checkpoints.

Or you can create a soft link to it.

```
grok-1/
├── checkpoints
│   └── ckpt-0 -> /your_data_path/ckpt-0
├── ...
```

## 4. Copy files & Run
```bash
cp prompt.json inference.py grok-1
python grok-1/inference.py
```
| **Parameter** | **Default Value** |
| :---: | :--- |
| **input-tokens** | 32 |
| **max-new-tokens** | 32 |
| **num-iter** | 4 |
| **num-warmup** | 1 |
