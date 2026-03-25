import argparse
import json
import os
import random
import yaml
import torch
from typing import Optional

from datasets import Dataset
from utils import io_utils
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_forget_rows(instances: list) -> list:
    """
    Forget set: replace original assistant response with Y' (tool-free response).
    Mirrors TOOLDELETE-SFT §3.4: for (ti, Qi, Yi) ∈ Tf, use Y'i instead of Yi.
    """
    rows = []
    skipped = 0
    for inst in instances:
        # support both flat fields and nested messages
        if "messages" in inst and inst["messages"]:
            user_content = inst["messages"][0].get("content", "").strip()
        else:
            user_content = inst.get("input", inst.get("user_input", "")).strip()

        y_prime = inst.get("y_prime", "").strip()

        if not user_content or not y_prime:
            skipped += 1
            continue

        rows.append(
            {
                "Name": inst.get("Name", inst.get("tool_name", "")),
                "instance_id": inst.get("instance_id", ""),
                "split": "forget",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": y_prime},
                ],
            }
        )

    print(f"  Forget rows built: {len(rows)}  (skipped {skipped} missing y_prime)")
    return rows


def build_retain_rows(instances: list) -> list:
    """
    Retain set: keep original query + original trace.
    Mirrors TOOLDELETE-SFT §3.2: re-expose model to Tr demonstrations.
    """
    rows = []
    skipped = 0
    for inst in instances:
        if "messages" in inst and len(inst["messages"]) >= 2:
            user_content = inst["messages"][0].get("content", "").strip()
            asst_content = inst["messages"][1].get("content", "").strip()
        else:
            user_content = inst.get("input", inst.get("user_input", "")).strip()
            asst_content = inst.get("trace_text", inst.get("gt_trace", "")).strip()

        if not user_content or not asst_content:
            skipped += 1
            continue

        rows.append(
            {
                "Name": inst.get("Name", inst.get("tool_name", "")),
                "instance_id": inst.get("instance_id", ""),
                "split": "retain",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": asst_content},
                ],
            }
        )

    print(f"  Retain rows built: {len(rows)}  (skipped {skipped} missing content)")
    return rows


def balance_and_merge(
    forget_rows: list,
    retain_rows: list,
    balance_ratio: float = 1.0,
    seed: int = 42,
) -> list:
    """
    Subsample retain to balance_ratio × |forget|, then merge and shuffle.
    Paper uses retain subset proportional to Tf for efficiency (§3.2).
    """
    rnd = random.Random(seed)
    n_target = int(len(forget_rows) * balance_ratio)
    if n_target < len(retain_rows):
        retain_rows = rnd.sample(retain_rows, n_target)
        print(f"  Retain subsampled to {len(retain_rows)} (ratio={balance_ratio})")

    combined = forget_rows + retain_rows
    rnd.shuffle(combined)
    print(
        f"  Total training rows: {len(combined)} "
        f"(forget={len(forget_rows)}, retain={len(retain_rows)})"
    )
    return combined


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        local_files_only=True,
        legacy=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for SFT causal LM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    return tokenizer, model


def apply_task_arithmetic(
    trained_model: AutoModelForCausalLM,
    base_model_path: str,  # θ0: instruction-tuned LLM (e.g. Vicuna)
    random_model_path: Optional[str],  # θR: randomly-initialized LLM (optional)
    alpha: float = 1.0,
    device: str = "cpu",
) -> AutoModelForCausalLM:
    """
    Apply task arithmetic to restore general capabilities lost during unlearning.
    If random_model_path is None, skips task arithmetic (no-op).
    """
    if random_model_path is None:
        print("  [Task Arithmetic] random_model_path not provided — skipping.")
        return trained_model

    print(f"  [Task Arithmetic] Loading θ0 from {base_model_path} ...")
    theta0 = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        local_files_only=True,
        device_map=device,
    )

    print(f"  [Task Arithmetic] Loading θR from {random_model_path} ...")
    thetaR = AutoModelForCausalLM.from_pretrained(
        random_model_path,
        torch_dtype=torch.float32,
        local_files_only=True,
        device_map=device,
    )

    print(f"  [Task Arithmetic] Applying α={alpha} × (θ0 − θR) to trained model ...")
    with torch.no_grad():
        for name, param in trained_model.named_parameters():
            if name in dict(theta0.named_parameters()) and name in dict(
                thetaR.named_parameters()
            ):
                p0 = dict(theta0.named_parameters())[name].to(param.device).float()
                pR = dict(thetaR.named_parameters())[name].to(param.device).float()
                delta = alpha * (p0 - pR)
                param.data = param.data.float() + delta
                param.data = param.data.to(torch.float16)

    del theta0, thetaR
    torch.cuda.empty_cache()
    print("  [Task Arithmetic] Done.")
    return trained_model


def parse_args():
    ap = argparse.ArgumentParser(description="TOOLDELETE-SFT training script")
    ap.add_argument(
        "--config",
        default=None,
        help="Path to base.yaml (optional; CLI args take priority)",
    )

    # data
    ap.add_argument(
        "--forget_jsonl",
        default=None,
        help="JSONL with forget instances + y_prime field",
    )
    ap.add_argument(
        "--retain_jsonl",
        default=None,
        help="JSONL with retain instances (original traces)",
    )
    ap.add_argument(
        "--balance_ratio",
        type=float,
        default=1.0,
        help="retain / forget size ratio (default 1.0 = equal)",
    )

    # models
    ap.add_argument(
        "--model_path", default=None, help="Tool-augmented model to unlearn (f)"
    )
    ap.add_argument(
        "--base_model",
        default=None,
        help="Instruction-tuned base model θ0 for task arithmetic",
    )
    ap.add_argument(
        "--random_model",
        default=None,
        help="Randomly-initialized model θR for task arithmetic (optional)",
    )
    ap.add_argument(
        "--ta_alpha",
        type=float,
        default=1.0,
        help="Task arithmetic scale α (default 1.0)",
    )

    # training
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate (paper uses 1e-5)"
    )
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--bf16", action="store_true", help="Use bf16 (A100+). Otherwise fp16."
    )

    return ap.parse_args()


def main():
    args = parse_args()

    cfg = {}
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)
        print(f"Loaded config from {args.config}")

    def get(cli_val, cfg_key, default=None):
        return cli_val if cli_val is not None else cfg.get(cfg_key, default)

    forget_jsonl = get(args.forget_jsonl, "yprime_out", "out/yprime_out.jsonl")
    retain_jsonl = get(args.retain_jsonl, "retain_sft", "out/retain_sft.jsonl")
    model_path = get(args.model_path, "toolalpaca_model", "TangQiaoYu/ToolAlpaca-7B")
    base_model = get(args.base_model, "base_model", None)
    random_model = get(args.random_model, "random_model", None)
    output_dir = get(args.output_dir, "out_dir", "out/tooldelete_sft")
    output_dir = os.path.join(output_dir, "tooldelete_sft_model")

    print("=" * 70)
    print("TOOLDELETE-SFT  —  Tool Unlearning for Tool-Augmented LLMs")
    print("=" * 70)
    print(f"  forget_jsonl : {forget_jsonl}")
    print(f"  retain_jsonl : {retain_jsonl}")
    print(f"  model_path   : {model_path}")
    print(f"  output_dir   : {output_dir}")
    print(
        f"  lr={args.lr}  epochs={args.epochs}  "
        f"batch={args.batch_size}  grad_accum={args.grad_accum}"
    )

    print("\n[1/4] Loading data ...")
    forget_instances = io_utils.read_jsonl(forget_jsonl)
    retain_instances = io_utils.read_jsonl(retain_jsonl)
    print(f"  Raw forget instances : {len(forget_instances)}")
    print(f"  Raw retain instances : {len(retain_instances)}")

    forget_rows = build_forget_rows(forget_instances)
    retain_rows = build_retain_rows(retain_instances)
    train_rows = balance_and_merge(
        forget_rows,
        retain_rows,
        balance_ratio=args.balance_ratio,
        seed=args.seed,
    )

    # convert to HuggingFace Dataset
    # SFTTrainer with `messages` field uses apply_chat_template automatically
    dataset = Dataset.from_list(train_rows)

    # ── load model ──
    print("\n[2/4] Loading model and tokenizer ...")
    tokenizer, model = load_model_and_tokenizer(model_path)

    # ── train ──
    print("\n[3/4] Training (TOOLDELETE-SFT) ...")
    io_utils.ensure_dir(output_dir)

    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=args.bf16,
        fp16=not args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        max_seq_length=args.max_seq_len,
        seed=args.seed,
        report_to="none",
        # use messages field directly
        dataset_text_field=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_cfg,
    )

    trainer.train()

    # save intermediate (before task arithmetic)
    pre_ta_dir = os.path.join(output_dir, "pre_task_arithmetic")
    trainer.save_model(pre_ta_dir)
    tokenizer.save_pretrained(pre_ta_dir)
    print(f"  Saved pre-task-arithmetic model → {pre_ta_dir}")

    # ── task arithmetic (GCR) ──
    print("\n[4/4] Applying Task Arithmetic for General Capability Retention ...")
    model = apply_task_arithmetic(
        trained_model=trainer.model,
        base_model_path=base_model if base_model else model_path,
        random_model_path=random_model,
        alpha=args.ta_alpha,
    )

    final_dir = os.path.join(output_dir, "final")
    io_utils.ensure_dir(final_dir)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n✓ Final model saved → {final_dir}")

    # cleanup
    del model
    torch.cuda.empty_cache()
    print("VRAM released.")

    print("\n" + "=" * 70)
    print("Done! Evaluate with:")
    print(f"  python eval_generate.py --config configs/base.yaml")
    print(f"  (point toolalpaca_model → {final_dir})")
    print("=" * 70)


if __name__ == "__main__":
    main()
