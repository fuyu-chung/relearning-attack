import re
import json
import yaml
import numpy as np
import torch
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from utils.io_utils import ensure_dir, read_jsonl, write_json


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_judge(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        local_files_only=True,
        legacy=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()

    judge_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return judge_pipe


def judge_tool_use(
    judge_pipe, pred_trace: str, gt_trace: str, instruction: str
) -> bool:
    messages = [
        {
            "role": "system",
            "content": (
                "You are evaluating whether an AI assistant's solution correctly "
                "calls the same tool as the standard answer.\n\n"
                "Follow these guidelines:\n"
                "1. The standard answer only includes function names and parameters, "
                "while the AI assistant's solution also includes function returns. "
                "It is acceptable to call getDetails, retrievalDataFromFile, or retry a function based on the return. "
                "These are not considered unrelated calls.\n"
                "2. Random calls to unrelated functions are not allowed.\n"
                "3. The solution must contain all the steps in the standard answer.\n"
                "4. The necessary parameters of the function need to be consistent with the standard answer. "
                "Parameters not mentioned in the task instruction can be inconsistent.\n"
                "5. The Observation and Final Answer do not need to match the standard answer, "
                "but should be reasonable and consistent with the call process.\n"
                "6. First provide a brief analysis, then give your answer.\n\n"
                "Output format:\n"
                "## Analysis\n"
                "{some analysis}\n"
                "## Result\n"
                "Process Correctness: one of [Yes, No]"
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Task Instruction\n{instruction}\n\n"
                f"## Standard Answer\n{gt_trace}\n\n"
                f"## AI Assistant's Solution\n{pred_trace}\n\n"
                f"## Analysis"
            ),
        },
    ]

    result = judge_pipe(
        messages,
        max_new_tokens=500,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=judge_pipe.tokenizer.eos_token_id,
    )
    new_text = result[0]["generated_text"][-1]["content"].strip()
    print(f"\nJudge: {new_text}\n")

    match = re.search(r"Process Correctness:\s*(Yes|No)", new_text, re.IGNORECASE)
    if match:
        correct = match.group(1).lower() == "yes"
        return correct, new_text

    # fallback
    gt_actions = re.findall(r"Action:\s*(\S+)", gt_trace)
    correct = any(action in pred_trace for action in gt_actions)
    return correct, new_text


def main():
    ap = ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["out_dir"])

    traces = read_jsonl(cfg["output_file"])
    print(f"Loaded {len(traces)} instances from {cfg['output_file']}")

    judge_pipe = load_judge(cfg["judge_model"])

    scores_path = Path(cfg["out_dir"]) / "eval_scores.jsonl"
    done_ids = set()
    if scores_path.exists():
        done = read_jsonl(str(scores_path))
        done_ids = {r["instance_id"] for r in done}
        print(f"Resuming: {len(done_ids)} already done")

    print(f"\nStart judging ({len(traces)} samples)...")

    false_reasons_path = Path(cfg["out_dir"]) / "eval_false_reasons.jsonl"

    with open(scores_path, "a", encoding="utf-8") as f, open(
        false_reasons_path, "a", encoding="utf-8"
    ) as fr:
        for item in tqdm(traces, desc="Judging"):
            instance_id = item["instance_id"]
            if instance_id in done_ids:
                continue

            print(f"\n[{instance_id}] {item.get('tool_name', '')}")

            try:
                used, reason = judge_tool_use(
                    judge_pipe,
                    item["pred_trace"],
                    item["gt_trace"],
                    item["input"],
                )
                row = {
                    "instance_id": instance_id,
                    "tool_name": item["tool_name"],
                    "split": item.get("split", "unknown"),
                    "used_tool": used,
                }
                if not used:
                    fr.write(
                        json.dumps(
                            {
                                "instance_id": instance_id,
                                "tool_name": item["tool_name"],
                                "split": item.get("split", "unknown"),
                                "reason": reason,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    fr.flush()
            except Exception as e:
                print(f"Skip {instance_id}: {e}")
                row = {
                    "instance_id": instance_id,
                    "tool_name": item.get("tool_name", "unknown"),
                    "split": item.get("split", "unknown"),
                    "used_tool": False,
                }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    all_scores = read_jsonl(str(scores_path))
    df = pd.DataFrame(all_scores)

    Tt = df["used_tool"].mean()
    Tf = df[df["split"] == "forget"]["used_tool"].mean()
    Tr = df[df["split"] == "retain"]["used_tool"].mean()

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"{'Metric':<8} {'Score':<12} {'Baseline':<12}")
    print("-" * 60)
    print(f"{'Tt':<8} {Tt:<12.3f} {'60.0':<12}")
    print(f"{'Tf':<8} {Tf:<12.3f} {'75.7':<12}")
    print(f"{'Tr':<8} {Tr:<12.3f} {'73.1':<12}")
    print("=" * 60)

    write_json(
        str(Path(cfg["out_dir"]) / "evaluation_summary.json"),
        {
            "Tt": round(float(Tt), 4),
            "Tf": round(float(Tf), 4) if not np.isnan(Tf) else None,
            "Tr": round(float(Tr), 4) if not np.isnan(Tr) else None,
            "n_total": len(df),
            "n_forget": int((df["split"] == "forget").sum()),
            "n_retain": int((df["split"] == "retain").sum()),
            "n_test": int((df["split"] == "test").sum()),
        },
    )

    print(f"\nSaved to: {cfg['out_dir']}/evaluation_summary.json")


if __name__ == "__main__":
    main()
