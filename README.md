# faast
Code base for [FAAST: Forward-Only Associative Learning via Closed-Form Fast Weights for Test-Time Supervised Adaptation](https://arxiv.org/pdf/2605.04651)

## Pretrained LLMs and Their Usage 

* 🔗[gpt2-xl, Qwen2.5-3B-Instruct, and Qwen2.5-7B-Instruct](https://huggingface.co/collections/gshbao/faast)  that can learn at test time.

* How to use these models?
```
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

fewshot_samples = ['sample 1', 'sample 2', ...]
inputs = tokenizer(fewshot_samples, return_tensors="pt", padding=True)

model.reset_projection() # clear existing fast weights
model.learn(**inputs)  # learn new fast weights
model.generate(...)  # do the task using the learned fast weights
```

## Installation instructions

First, create a Python virtual environment using `uv`:
> [!TIP]
> To install `uv`, follow the [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).
```shell
uv venv handbook --python 3.11 && source handbook/bin/activate && uv pip install --upgrade pip
```

Next, install PyTorch `v2.6.0`

```shell
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
uv pip install "flash-attn==2.7.4.post1" --no-build-isolation
```

You can then install the remaining package dependencies as follows:

```shell
uv pip install .
```


Last, log into your Hugging Face account and download the pretrained models:

```shell
huggingface-cli login
huggingface-cli download gshbao/faast-gpt2-xl  --local-dir ./faast_models/faast-gpt2-xl
huggingface-cli download gshbao/faast-Qwen2.5-3B-Instruct  --local-dir ./faast_models/faast-Qwen2.5-3B-Instruct
huggingface-cli download gshbao/faast-Qwen2.5-7B-Instruct  --local-dir ./faast_models/faast-Qwen2.5-7B-Instruct
```


## Project structure

```
├── LICENSE
├── llm/                        <- The implementation of the FAAST module for LLMs
├── README.md                   <- The top-level README for developers using this project
├── scripts/                    <- Scripts to train and evaluate LLMs extended with FAAST module.
├── setup.cfg                   <- Installation config
├── setup.py                    <- Makes project pip installable (pip install -e .) 
└── vision/                     <- The implementation of the FAAST module for image classification
```


## Citation

If you find the content of this repo useful, please cite it as follows:

```bibtex
@article{bao2026faast,
  title={FAAST: Forward-Only Associative Learning via Closed-Form Fast Weights for Test-Time Supervised Adaptation},
  author={Bao, Guangsheng and Zhang, Hongbo and Cui, Han and Sun, Ke and Zhao, Yanbin and He, Juncai and Zhang, Yue},
  journal={arXiv preprint arXiv:2605.04651},
  year={2026}
}
```
