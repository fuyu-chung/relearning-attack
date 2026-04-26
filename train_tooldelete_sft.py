import argparse
import os
import random
import torch

from collections import defaultdict
from typing import Optional

from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model

from utils import io_utils
from utils.io_utils import load_config, resolve_config_key, read_json, read_jsonl
from utils.io_utils import load_model as _load_model_shared
from utils.trace_utils import build_forget_rows, build_retain_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ToolDelete SFT training")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--suffix", default=None, help="Append suffix to output_dir")
    ap.add_argument("--forget_data_path", default=None)
    ap.add_argument("--retain_data_path", default=None)
    ap.add_argument("--model_output_dir", default=None)
    ap.add_argument("--retain_ratio", default=None, type=float)
    return ap.parse_args()


def normalize_suffix(raw_suffix: Optional[str]) -> str:
    if not raw_suffix:
        return ""
    return raw_suffix if raw_suffix.startswith("_") else f"_{raw_suffix}"


def append_suffix_to_dir(path_str: str, suffix: str) -> str:
    if not suffix:
        return path_str
    path = path_str.rstrip("/")
    return path if path.endswith(suffix) else f"{path}{suffix}"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_input_file(path: str, label: str) -> None:
    if not path:
        raise ValueError(f"{label} path is empty.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def load_model(model_path: str, local_files_only: bool = True):
    """Thin wrapper around the shared load_model that sets training defaults."""
    tokenizer, model = _load_model_shared(model_path, local_files_only=local_files_only)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.unk_token
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return tokenizer, model


def save_model_bundle(model, tokenizer, output_dir: str) -> None:
    io_utils.ensure_dir(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def stratified_sample_by_tool(rows: list, target_count: int, seed: int = 42) -> list:
    if target_count <= 0:
        return []
    if target_count >= len(rows):
        rnd = random.Random(seed)
        sampled = list(rows)
        rnd.shuffle(sampled)
        return sampled

    groups = defaultdict(list)
    for row in rows:
        groups[row.get("Name", "")].append(row)

    rnd = random.Random(seed)
    allocations = {}
    fractions = []

    for name, group_rows in groups.items():
        exact = len(group_rows) * target_count / len(rows)
        base = min(len(group_rows), int(exact))
        allocations[name] = base
        fractions.append((exact - base, name))

    assigned = sum(allocations.values())
    remaining = target_count - assigned
    fractions.sort(reverse=True)

    for _, name in fractions:
        if remaining <= 0:
            break
        if allocations[name] < len(groups[name]):
            allocations[name] += 1
            remaining -= 1

    if remaining > 0:
        names = list(groups.keys())
        rnd.shuffle(names)
        for name in names:
            if remaining <= 0:
                break
            available = len(groups[name]) - allocations[name]
            if available <= 0:
                continue
            take = min(available, remaining)
            allocations[name] += take
            remaining -= take

    sampled = []
    for name, group_rows in groups.items():
        take = allocations[name]
        if take > 0:
            sampled.extend(rnd.sample(group_rows, take))
    rnd.shuffle(sampled)
    return sampled


def balance_and_merge(
    forget_rows: list,
    retain_rows: list,
    balance_ratio: float = 2.0,
    seed: int = 42,
    retain_sampling_mode: str = "random",
) -> list:
    if balance_ratio < 0:
        raise ValueError(f"balance_ratio must be >= 0, got {balance_ratio}")
    if not forget_rows:
        raise ValueError("No valid forget rows found. Check forget_yprime generation.")

    rnd = random.Random(seed)
    n_target = int(len(forget_rows) * balance_ratio)
    if n_target < len(retain_rows):
        if retain_sampling_mode == "stratified":
            retain_rows = stratified_sample_by_tool(retain_rows, n_target, seed)
        else:
            retain_rows = rnd.sample(retain_rows, n_target)

    combined = forget_rows + retain_rows
    if not combined:
        raise ValueError("No training rows available after merge.")
    rnd.shuffle(combined)
    return combined


IGNORE_TOKEN_ID = LabelSmoother.ignore_index
EOS_TOKEN = "</s>"


def preprocess(sources, tokenizer) -> dict:
    """Token-level masking: only trainable=True segments contribute to loss.

    Mirrors train.py preprocess() exactly.
    """
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


class ToolDeleteDataset(Dataset):
    """Lazy dataset for ToolDelete SFT, mirrors LazySupervisedDataset from train.py."""

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


def ensure_random_model(model_path: str, output_path: str, seed: int = 42) -> str:
    """Generate a random-initialised checkpoint (θᴿ) for task arithmetic."""
    if os.path.exists(os.path.join(output_path, "config.json")):
        print(f"✓ Random model already exists at {output_path}")
        return output_path

    print(f"Generating random model from {model_path}...")
    torch.manual_seed(seed)
    random.seed(seed)

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    prev_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        random_model = AutoModelForCausalLM.from_config(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    io_utils.ensure_dir(output_path)
    random_model.save_pretrained(output_path)
    print(f"✓ Random model saved to {output_path}")
    del random_model
    torch.cuda.empty_cache()
    return output_path


def apply_task_arithmetic(
    trained_model: AutoModelForCausalLM,
    base_model: str,
    random_model_path: Optional[str],
    alpha: float = 0.5,
    local_files_only: bool = True,
) -> AutoModelForCausalLM:
    if random_model_path is None:
        return trained_model

    load_kw = {
        "torch_dtype": torch.float16,
        "device_map": "cpu",
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }

    theta0 = AutoModelForCausalLM.from_pretrained(base_model, **load_kw)
    theta0_params = dict(theta0.named_parameters())
    thetaR = AutoModelForCausalLM.from_pretrained(random_model_path, **load_kw)
    thetaR_params = dict(thetaR.named_parameters())

    with torch.no_grad():
        for name, param in trained_model.named_parameters():
            if name in theta0_params and name in thetaR_params:
                device = param.device
                delta = theta0_params[name].to(
                    device=device, dtype=param.dtype, non_blocking=True
                )
                delta.sub_(
                    thetaR_params[name].to(
                        device=device, dtype=param.dtype, non_blocking=True
                    )
                )
                param.data.add_(delta, alpha=alpha)
                del delta

    del theta0, theta0_params, thetaR, thetaR_params
    torch.cuda.empty_cache()
    return trained_model


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    cfg = load_config(args.config)
    suffix = normalize_suffix(
        args.suffix if args.suffix is not None else cfg.get("path_suffix")
    )

    out_root = resolve_config_key(
        cfg, "output_dir", "out_dir", required=False, default="out"
    )

    forget_yprime_path = resolve_config_key(
        cfg,
        "forget_data_path",
        "forget_yprime",
        required=False,
        default="out/forget_yprime.jsonl",
    )
    # retain_train.json is the canonical output from prep_train.py
    retain_json = resolve_config_key(
        cfg,
        "retain_data_path",
        required=False,
        default="out/retain_train.json",
    )
    model_path = resolve_config_key(
        cfg,
        "model_path",
        "train_model_path",
        "base_model_path",
        "base_model",
        "toolalpaca_model",
        required=False,
        default="TangQiaoYu/ToolAlpaca-7B",
    )
    output_dir = resolve_config_key(
        cfg,
        "model_output_dir",
        "tooldelete_sft_model_dir",
        required=False,
        default=os.path.join(out_root, "tooldelete_sft_model"),
    )
    random_model = cfg.get("random_model_path", cfg.get("random_model"))

    if args.forget_data_path:
        forget_yprime_path = args.forget_data_path
    if args.retain_data_path:
        retain_json = args.retain_data_path
    if args.model_output_dir:
        output_dir = args.model_output_dir
    if suffix:
        output_dir = append_suffix_to_dir(output_dir, suffix)

    retain_balance_ratio = float(
        cfg.get("retain_ratio", cfg.get("retain_balance_ratio", 1.0))
    )
    retain_sampling_mode = str(
        cfg.get("retain_sample_mode", cfg.get("retain_sampling_mode", "stratified"))
    )
    task_arithmetic_alpha = float(
        cfg.get("ta_alpha", cfg.get("task_arithmetic_alpha", 1.0))
    )
    if args.retain_ratio is not None:
        retain_balance_ratio = args.retain_ratio

    set_global_seed(42)
    validate_input_file(forget_yprime_path, "forget_yprime")

    # forget rows: from gen_yprime output (Name, instance_id, question, y_prime)
    forget_rows = build_forget_rows(list(read_jsonl(forget_yprime_path)))

    # retain rows: from retain_train.json flat records (Name, instance_id, process, trainable)
    retain_rows = build_retain_rows(read_json(retain_json))

    train_rows = balance_and_merge(
        forget_rows,
        retain_rows,
        retain_balance_ratio,
        42,
        retain_sampling_mode,
    )

    tokenizer, model = load_model(model_path, local_files_only=True)

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

    train_dataset = ToolDeleteDataset(train_rows, tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=False,
            fp16=True,
            optim="adafactor",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            seed=42,
            report_to="none",
        ),
    )
    trainer.train()

    pre_ta_dir = os.path.join(output_dir, "pre_task_arithmetic")
    save_model_bundle(trainer.model, tokenizer, pre_ta_dir)

    print("\n=== Preparing task arithmetic ===")
    if random_model is None:
        random_model = os.path.join(out_root, "random_model")
    random_model = ensure_random_model(model_path, random_model, seed=42)
    print(f"✓ Random model ready: {random_model}\n")

    model = (
        trainer.model.merge_and_unload()
        if hasattr(trainer.model, "merge_and_unload")
        else trainer.model
    )
    model = apply_task_arithmetic(
        trained_model=model,
        base_model=model_path,
        random_model_path=random_model,
        alpha=task_arithmetic_alpha,
        local_files_only=True,
    )

    final_dir = os.path.join(output_dir, "final")
    save_model_bundle(model, tokenizer, final_dir)
    print(f"Final model saved: {final_dir}")

    del model
    torch.cuda.empty_cache()
    print("Done")


if __name__ == "__main__":
    main()
