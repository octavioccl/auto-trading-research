from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from llm_trading.common import (
    DEFAULT_QWEN_MODEL_ID,
    TRADING_SYSTEM_PROMPT,
    default_llm_artifact_dir,
    default_llm_dataset_path,
    ensure_llm_dirs,
    parse_completion_text,
    preferred_compute_dtype,
)


def _require_llm_dependencies():
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing LLM training dependencies. Install transformers, peft, bitsandbytes, accelerate, trl, and datasets."
        ) from exc


class SFTDataset(Dataset):
    def __init__(self, records: List[Dict[str, str]], tokenizer, max_seq_length: int):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        for record in records:
            tokenized = self._tokenize(record["prompt_text"], record["completion_text"])
            if tokenized is not None:
                self.examples.append(tokenized)

    def _chat_text(self, prompt_text: str, completion_text: str | None = None) -> str:
        messages = [
            {"role": "system", "content": TRADING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        if completion_text is not None:
            messages.append({"role": "assistant", "content": completion_text})
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=completion_text is None,
            )
        suffix = "" if completion_text is None else completion_text
        return f"{TRADING_SYSTEM_PROMPT}\n\n{prompt_text}\n\n{suffix}"

    def _tokenize(self, prompt_text: str, completion_text: str) -> Dict[str, List[int]] | None:
        prompt_only = self._chat_text(prompt_text, None)
        full_text = self._chat_text(prompt_text, completion_text)
        prompt_ids = self.tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        full_ids = full_ids[: self.max_seq_length]
        if len(prompt_ids) >= len(full_ids):
            return None
        prompt_len = min(len(prompt_ids), len(full_ids) - 1)
        labels = full_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len
        attention_mask = [1] * len(full_ids)
        return {"input_ids": full_ids, "labels": labels, "attention_mask": attention_mask}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]


class DataCollatorForCausalJSON:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_masks = []
        labels = []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad)
            attention_masks.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def infer_lora_target_modules(model) -> List[str]:
    target_modules = set()
    for name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if "linear" not in class_name:
            continue
        leaf = name.split(".")[-1]
        if leaf == "lm_head":
            continue
        target_modules.add(leaf)
    if not target_modules:
        raise ValueError("Unable to infer LoRA target modules for the selected model.")
    return sorted(target_modules)


def _load_dataset_records(dataset_path: Path, max_train_samples: int, max_eval_samples: int) -> tuple[List[Dict[str, str]], List[Dict[str, str]], dict[str, Any]]:
    frame = pd.read_parquet(dataset_path)
    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    val_frame = frame[frame["split"] == "val"].reset_index(drop=True)
    if max_train_samples > 0:
        train_frame = train_frame.head(max_train_samples)
    if max_eval_samples > 0:
        val_frame = val_frame.head(max_eval_samples)
    metadata = {
        "dataset_path": str(dataset_path),
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
    }
    return train_frame.to_dict("records"), val_frame.to_dict("records"), metadata


def train_llm(args: argparse.Namespace) -> Dict[str, Any]:
    _require_llm_dependencies()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    ensure_llm_dirs()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_train, records_val, dataset_metadata = _load_dataset_records(
        dataset_path, args.max_train_samples, args.max_eval_samples
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = SFTDataset(records_train, tokenizer, args.max_seq_length)
    val_dataset = SFTDataset(records_val, tokenizer, args.max_seq_length)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=preferred_compute_dtype(),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=infer_lora_target_modules(model),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=preferred_compute_dtype() == torch.float16,
        bf16=preferred_compute_dtype() == torch.bfloat16,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForCausalJSON(tokenizer),
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "base_model": args.base_model,
        "dataset": dataset_metadata,
        "output_dir": str(output_dir),
        "max_seq_length": args.max_seq_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    summary_path = default_llm_artifact_dir() / "evals" / "latest_train_llm_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA supervised fine-tuning for the trading LLM.")
    parser.add_argument("--dataset-path", default=str(default_llm_dataset_path()))
    parser.add_argument("--output-dir", default=str(default_llm_artifact_dir() / "adapters" / "qwen35_9b_lora"))
    parser.add_argument("--base-model", default=DEFAULT_QWEN_MODEL_ID)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    args = parser.parse_args()
    train_llm(args)


if __name__ == "__main__":
    main()
