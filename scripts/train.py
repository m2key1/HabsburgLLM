import json
import math
import os
import time
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from datasets import DatasetDict, load_from_disk
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Olmo3Config,
    Olmo3ForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

dataset_name = "habsburg"

tokenized_path = Path("data/tokenized") / dataset_name  # produced by notebooks/tokenize_corpus.ipynb
tokenizer_path = Path("data/tokenizer/habsburg-48k")
output_dir = Path("checkpoints") / f"olmo-{dataset_name}"

sequence_length = 4096  # must match tokenize_corpus.ipynb
seed = 42
lr_schedule = "cosine"  # or "wsd"


def rank0_print(*args, **kwargs):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


def build_model(tokenizer) -> Olmo3ForCausalLM:
    config = Olmo3Config(
        vocab_size=len(tokenizer),
        hidden_size=3072,
        intermediate_size=9728,
        num_hidden_layers=32,
        num_attention_heads=24,     # head_dim 128
        num_key_value_heads=8,      # GQA 3:1, smaller KV cache
        max_position_embeddings=sequence_length,
        sliding_window=sequence_length,  # == context, so full attention
        # required, else Olmo3 stock vocab
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    return Olmo3ForCausalLM(config)


def strip_compile_prefix(model_dir: Path):
    """Rewrite exported weights without torch.compile's _orig_mod. prefix, else from_pretrained loads nothing."""
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt") as handle:
            tensors = {key.removeprefix("_orig_mod."): handle.get_tensor(key) for key in handle.keys()}
        save_file(tensors, path, metadata={"format": "pt"})

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        index["weight_map"] = {k.removeprefix("_orig_mod."): v for k, v in index["weight_map"].items()}
        index_path.write_text(json.dumps(index, indent=2))


def preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to argmax token ids so the eval loop never gathers full logits."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    """Token-level top-1 accuracy over non-padding positions."""
    preds = eval_pred.predictions[:, :-1]
    labels = eval_pred.label_ids[:, 1:]
    mask = labels != -100
    return {"token_accuracy": float((preds == labels)[mask].mean())}


class MetricsTrainer(Trainer):
    """Trainer that logs derived metrics: perplexity, bits/token, tokens seen, throughput, weight norm."""

    def __init__(self, *args, tokens_per_step: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens_per_step = tokens_per_step
        self._last_time = None
        self._last_step = 0

    def _weight_norm(self) -> float:
        # under FSDP each rank holds disjoint shards, so local sum + all_reduce is exact
        with torch.no_grad():
            sq = torch.stack([p.detach().float().pow(2).sum() for p in self.model.parameters()]).sum()
        if dist.is_initialized():
            dist.all_reduce(sq)
        return sq.sqrt().item()

    def log(self, logs, start_time=None):
        step = self.state.global_step
        if "loss" in logs:
            logs["tokens_seen"] = step * self.tokens_per_step
            if self._last_time is not None and step > self._last_step:
                elapsed = time.perf_counter() - self._last_time
                logs["tokens_per_second"] = (step - self._last_step) * self.tokens_per_step / elapsed
            logs["weight_norm"] = self._weight_norm()
        for key in [k for k in logs if k.endswith("_loss")]:
            logs[key.replace("_loss", "_perplexity")] = math.exp(logs[key])
            logs[key.replace("_loss", "_bits_per_token")] = logs[key] / math.log(2)
        # timer resets on eval logs too, so eval time never counts into throughput
        self._last_time = time.perf_counter()
        self._last_step = step
        super().log(logs, start_time)


def main():
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = cast(DatasetDict, load_from_disk(str(tokenized_path)))

    eval_sets = {"val": dataset["test"]}
    extra_eval_dir = Path("data/tokenized/eval")
    if extra_eval_dir.is_dir():
        for path in sorted(p for p in extra_eval_dir.iterdir() if p.is_dir()):
            eval_sets[path.name] = load_from_disk(str(path))

    rank0_print(f"Train sequences: {len(dataset['train'])}")
    for name, eval_set in eval_sets.items():
        rank0_print(f"Eval sequences ({name}): {len(eval_set)}")
    rank0_print(f"Tokens trained:  {len(dataset['train']) * sequence_length:,}")

    model = build_model(tokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Model parameters:     {n_params / 1e6:.1f}M")
    rank0_print(f"Trainable parameters: {n_trainable / 1e6:.1f}M")

    # 5 epochs measured: the 5th bought only -0.013 eval loss over 2.8 days
    per_device_batch, grad_accum, epochs = 50, 3, 4
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    total_steps = math.ceil(len(dataset["train"]) / (per_device_batch * grad_accum * world_size)) * epochs
    scheduler = {
        "cosine": {"lr_scheduler_type": "cosine"},
        "wsd": {
            "lr_scheduler_type": "warmup_stable_decay",
            "lr_scheduler_kwargs": {
                "decay_type": "1-sqrt",
                "min_lr_ratio": 0.0,
                "num_decay_steps": total_steps // 10,
            },
        },
    }[lr_schedule]

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        dataloader_num_workers=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_compile=True,                    # +26% throughput and -35% memory, measured
        learning_rate=3e-4,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,                       # < 0.999 default, more stable
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        warmup_steps=2000,
        **scheduler,
        bf16=True,
        tf32=True,
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": ["Olmo3DecoderLayer"]},
        ddp_timeout=7200,  # the 1800 s default expired reloading the best model and killed the run

        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=100,
        load_best_model_at_end=True,
        logging_steps=50,
        logging_first_step=True,
        report_to="tensorboard",
        logging_dir=str(output_dir / "logs"),
        seed=seed,
    )

    tokens_per_step = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
        * args.world_size * sequence_length
    )

    trainer = MetricsTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        # plain dataset keeps the eval_loss metric names stable
        eval_dataset=eval_sets if len(eval_sets) > 1 else eval_sets["val"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        tokens_per_step=tokens_per_step,
    )

    checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    if checkpoint:
        rank0_print(f"Resuming from checkpoint: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output_dir / "final"))
    if trainer.is_world_process_zero():
        strip_compile_prefix(output_dir / "final")


if __name__ == "__main__":
    main()
