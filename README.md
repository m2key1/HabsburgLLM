# HabsburgLLM

A research project to train an LLM from scratch using only historical data. The goal is knowledge creation and research, not producing a production-quality language model. The focus is on learning what happens when a model is trained exclusively on historical texts.

## Models

**olmo-habsburg** is a 3.97B-parameter OLMo 3 model trained on Austrian newspapers from ANNO (Austrian National Library) published between 1800 and 27 June 1914, the day before Franz Ferdinand was assassinated in Sarajevo.

**olmo-london** is an earlier model trained on London texts from 1800 to 1875 ([postgrammar/london-llm-1800](https://huggingface.co/datasets/postgrammar/london-llm-1800)).

Weights are on Hugging Face at [M2key1/HabsburgLLM](https://huggingface.co/M2key1/HabsburgLLM). The final model is on `main`; intermediate checkpoints are separate revisions.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "M2key1/HabsburgLLM"
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate(prompt, subfolder=""):
    tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=subfolder)
    model = AutoModelForCausalLM.from_pretrained(repo, subfolder=subfolder, dtype=torch.bfloat16).to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        no_repeat_ngram_size=6,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate("Wien, 27. Juni."))                                    # olmo-habsburg
print(generate("London, June 1851.", subfolder="olmo-london/final"))  # olmo-london
```

It is a base model: it continues text, it does not answer questions. Greedy decoding loops; sample with `no_repeat_ngram_size=6` as in `notebooks/inference.ipynb`. Instruction fine-tuning is in the works. Coming Soon™.

## Data

The ANNO newspapers were re-OCRed to get clean text. The OCR output still had failure modes - repeated phrase loops and degenerate pages - which the cleaning step filters out. Of 1,198,378 documents across 115 years, 1,111,493 (92.7%) survive, about 103B characters or 22.9B tokens. The corpus itself is not distributed.

## Training run

olmo-habsburg: 32 layers, hidden size 3072, 24 attention heads with 8 KV heads, 4096-token context, 48k vocabulary. Five epochs over 22.9B tokens (114B tokens seen) on 8 NVIDIA H200 GPUs in 13.9 days, on the [DHinfra.at](https://www.dhinfra.at) cluster in Graz. Final validation loss 1.910 (perplexity 6.755) on a held-out split of the same corpus.

## Setup

Requires Python 3.12+ and a CUDA GPU. Dependencies are managed with [uv](https://docs.astral.sh/uv/); PyTorch comes from the CUDA 13.0 wheel index configured in `pyproject.toml`.

```bash
uv sync
uv run jupyter notebook notebooks/
```

## Pipeline

1. **Tokenizer** - `notebooks/train_tokenizer.ipynb`, 48k byte-level BPE.
2. **Tokenization** - `notebooks/tokenize_corpus.ipynb`, packs documents into 4096-token sequences.
3. **Training** - `scripts/train.py` via `sbatch train.slurm` (8 GPUs, FSDP, bf16).
4. **Inference** - `notebooks/inference.ipynb`.

## License

Code is MIT, see [LICENSE](LICENSE). Model weights on Hugging Face are Apache-2.0. The training data is not included.

## Citation

```bibtex
@software{fleischhacker_habsburgllm,
  author = {Fleischhacker, David},
  title  = {HabsburgLLM},
  year   = {2026},
  url    = {https://github.com/m2key1/HabsburgLLM}
}
```
