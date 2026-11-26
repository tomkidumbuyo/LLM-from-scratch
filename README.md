# GPT from Scratch

A small, educational GPT implementation in PyTorch. It includes a GPT-style
transformer, a text dataset loader, a training script, and a text-generation
command.

## Project layout

```text
.
├── data/                       # Local training text (not tracked by Git)
├── checkpoints/                # Saved model weights (not tracked by Git)
├── config.py                  # Shared paths and training/model configuration
├── finetuning/                 # Fine-tuning workflow scaffold
├── generation/
│   └── generate.py
├── pretraining/
│   ├── data.py
│   ├── train.py
│   └── transformer/
│       ├── attention.py
│       ├── feedforward.py
│       ├── gelu.py
│       ├── gpt.py
│       ├── layer_norm.py
│       └── transformer.py
├── .env.example
└── pyproject.toml
```

## Set up

Use Python 3.9 or newer. From the project root, create and activate a virtual
environment, then install the project and its dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Training data and configuration

Training reads plain text from `data/the-verdict.txt`. Add your own text file at
that path before training; datasets are intentionally excluded from Git.

Configuration defaults, including the model size, learning rate, and number of
epochs, are in `config.py`. Data-loader settings can be
overridden with environment variables. To use the example settings, copy
`.env.example` to `.env` and edit it as needed:

```bash
cp .env.example .env
```

`DATALOADER_MAX_LENGTH` must not exceed the model's `context_length` (256 by
default). `DATALOADER_STRIDE` controls how far the text window advances between
training examples.

## Train

From the project root, run:

```bash
python -m pretraining.train
```

The script selects CUDA, Apple Metal (MPS), or CPU when available, prints
training and validation losses, and saves the trained model to
`checkpoints/gpt_model.pth`. Training runs for 15 epochs by default and may be
slow on CPU.

## Generate text

After training, provide a prompt:

```bash
python -m generation.generate "Every effort moves you"
```

Useful options:

```bash
python -m generation.generate "Once upon a time" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-k 40
```

`--max-new-tokens` defaults to 50. A temperature of `0` (the default) uses
greedy decoding; a positive temperature samples from the model. `--top-k`
optionally limits sampling to the most likely tokens. Generation requires a
checkpoint produced by the training command.

Fine-tuning code has not been implemented yet; `finetuning/` is reserved for
that workflow. Training data, `.env`, and model checkpoints are local files and
are ignored by Git.
