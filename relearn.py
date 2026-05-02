"""relearn.py

用 forget_train.jsonl 對已 unlearn 的模型做 relearning attack，
觀察模型能學回多少 forget set 的工具能力 (Tf)。

等同於 train_tooldelete_sft.py 只保留 TKD 部分，拿掉 TKR (retain set) 和 GCR (task arithmetic)。
LoRA 設定與 train_tooldelete_sft.py 完全相同。

用法:
    python relearn.py --config configs/relearn.yaml
    python relearn.py --config configs/relearn.yaml --forget_samples 50
"""

import argparse
import os
import random
import torch

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, PeftModel

from utils.io_utils import load_config, resolve_config_key, read_jsonl, ensure_dir
from utils.io_utils import load_model as _load_model_shared
from utils.trace_utils import build_forget_rows

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
EOS_TOKEN = "</s>"


def preprocess(sources, tokenizer) -> dict:
    """與 train_tooldelete_sft.py 完全相同。"""
    conversations, trainables = [], []
    for source in sources:
        source[0][-1] += " " + EOS_TOKEN
        conversations.append(source[0])
        trainables.append(source[1])

    input_ids = tokenizer(
        ["".join(c) for c in conversations],
        return_tensors="pt",
        padding="max_length",
        max_length=2048,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    for conversation, target, trainable in zip(conversations, targets, trainables):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for conv, train in zip(conversation, trainable):
            round_len = len(tokenizer(conv).input_ids) - 2
            if conv.endswith(EOS_TOKEN):
                round_len += 1
            if not train:
                target[cur_len : cur_len + round_len] = IGNORE_TOKEN_ID
            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                print(f"WARNING: tokenization mismatch {cur_len} vs. {total_len}")

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class RelearnDataset(Dataset):
    def __init__(self, rows: list, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.list_data = [[r["process"], r["trainable"]] for r in rows]
        self.cached: dict = {}

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, i):
        if i in self.cached:
            return self.cached[i]
        ret = preprocess([self.list_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached[i] = ret
        return ret


def parse_args():
    ap = argparse.ArgumentParser(description="Relearning attack on unlearned model")
    ap.add_argument("--config", default="configs/relearn.yaml")
    ap.add_argument("--forget_samples", type=int, default=None)
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--forget_data_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_path = args.model_path or resolve_config_key(cfg, "model_path")
    base_model_path = cfg.get("base_model_path", None)
    forget_data_path = args.forget_data_path or resolve_config_key(
        cfg, "forget_data_path"
    )
    base_output_dir = args.output_dir or resolve_config_key(cfg, "output_dir")
    max_steps = args.max_steps or int(cfg.get("max_steps", 100))
    lr = float(cfg.get("lr", 1e-5))
    per_device_batch_size = int(cfg.get("per_device_batch_size", 1))
    gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 4))

    all_forget_rows = build_forget_rows(list(read_jsonl(forget_data_path)))
    print(f"Total forget rows: {len(all_forget_rows)}")
    if not all_forget_rows:
        raise ValueError("No forget rows found. Check forget_data_path.")

    # 決定抽幾筆
    forget_samples = args.forget_samples or cfg.get("forget_samples", None)
    if forget_samples is not None:
        forget_samples = int(forget_samples)
        if forget_samples > len(all_forget_rows):
            raise ValueError(
                f"forget_samples={forget_samples} 超過資料總數 {len(all_forget_rows)}"
            )
        random.seed(42)
        forget_rows = random.sample(all_forget_rows, forget_samples)
        output_dir = f"{base_output_dir}_{forget_samples}"
    else:
        forget_rows = all_forget_rows
        output_dir = f"{base_output_dir}_{len(forget_rows)}"

    print(f"Using {len(forget_rows)} forget rows -> {output_dir}")
    ensure_dir(output_dir)

    tokenizer, model = _load_model_shared(
        model_path, base_model_path=base_model_path, local_files_only=True
    )
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.unk_token
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # 若模型已經是 PeftModel（LoRA adapter）則直接 unfreeze adapter weights
    # 若是完整模型則加上新的 LoRA
    if isinstance(model, PeftModel):
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = RelearnDataset(forget_rows, tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=False,
            fp16=True,
            optim="adafactor",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            save_strategy="no",
            seed=42,
            report_to="none",
        ),
    )
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    ensure_dir(final_dir)
    model.merge_and_unload().save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved: {final_dir}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
