import re
import json
import numpy as np
import torch
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from utils.io_utils import (
    ensure_dir,
    read_jsonl,
    write_json,
    load_config,
    get_done_ids,
    resolve_config_key,
)


def load_judge(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        local_files_only=True,
        legacy=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=8,
    )


def strip_observation(trace: str) -> str:
    return re.sub(
        r"\n?\s*Observation:.*?(?=\n\s*(Action|Thought|Final Answer)|$)",
        "",
        trace,
        flags=re.DOTALL,
    ).strip()


def strip_duplicate_actions(trace: str) -> str:
    lines = trace.split("\n")
    cleaned = []
    i = 0
    while i < len(lines):
        cleaned.append(lines[i])
        if i + 2 < len(lines) and lines[i] == lines[i + 2]:
            i += 2
        else:
            i += 1
    return "\n".join(cleaned)


_JUDGE_SYSTEM = (
    "You are evaluating whether an AI assistant's solution correctly "
    "calls the same tool as the standard answer.\n\n"
    "Follow these guidelines:\n"
    "1. The standard answer only includes function names and parameters. "
    "It is acceptable to call getDetails or retrievalDataFromFile as additional steps. "
    "These are not considered unrelated calls.\n"
    "2. Random calls to unrelated functions are not allowed.\n"
    "3. The solution must contain all the steps in the standard answer.\n"
    "4. The necessary parameters of the function need to be consistent with "
    "the standard answer. Parameters not mentioned in the task instruction "
    "can be inconsistent.\n"
    "5. First provide a brief analysis, then give your answer.\n\n"
    "Output format:\n"
    "## Analysis\n"
    "{some analysis}\n"
    "## Result\n"
    "Process Correctness: one of [Yes, No]"
)


def judge_tool_use(
    judge_pipe, pred_trace: str, gt_trace: str, instruction: str
) -> tuple[bool, str]:
    gt_trace = strip_duplicate_actions(gt_trace)
    pred_trace = strip_observation(pred_trace)

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
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
        max_new_tokens=350,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=judge_pipe.tokenizer.eos_token_id,
    )
    new_text = result[0]["generated_text"][-1]["content"].strip()
    print(f"\nJudge: {new_text}\n")

    match = re.search(r"Process Correctness:\s*(Yes|No)", new_text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes", new_text

    gt_actions = re.findall(r"Action:\s*(\S+)", gt_trace)
    correct = any(action in pred_trace for action in gt_actions)
    return correct, new_text


def compute_and_print_summary(df: pd.DataFrame) -> dict:
    n_test = int((df["split"] == "test").sum())
    n_forget = int((df["split"] == "forget").sum())
    n_retain = int((df["split"] == "retain").sum())

    Tt = df[df["split"] == "test"]["used_tool"].mean()
    Tf = df[df["split"] == "forget"]["used_tool"].mean()
    Tr = df[df["split"] == "retain"]["used_tool"].mean()

    t_correct = int((df[df["split"] == "test"]["used_tool"] == True).sum())
    f_correct = int((df[df["split"] == "forget"]["used_tool"] == True).sum())
    r_correct = int((df[df["split"] == "retain"]["used_tool"] == True).sum())

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"{'Metric':<8} {'Score':<12} {'Baseline':<12} {'Sample Count':<15}")
    print("-" * 70)
    print(f"{'Tt':<8} {Tt:<12.3f} {'60.0':<12} {t_correct}/{n_test} ({Tt*100:6.2f}%)")
    print(
        f"{'Tf':<8} {Tf:<12.3f} {'75.7':<12} "
        f"{f_correct}/{n_forget} ({Tf*100 if not np.isnan(Tf) else 0:6.2f}%)"
    )
    if np.isnan(Tr):
        print(f"{'Tr':<8} {'N/A':<12} {'73.1':<12} {r_correct}/{n_retain} (no data)")
    else:
        print(
            f"{'Tr':<8} {Tr:<12.3f} {'73.1':<12} "
            f"{r_correct}/{n_retain} ({Tr*100:6.2f}%)"
        )
    print("=" * 70)

    return {
        "Tt": round(float(Tt), 4) if not np.isnan(Tt) else None,
        "Tf": round(float(Tf), 4) if not np.isnan(Tf) else None,
        "Tr": round(float(Tr), 4) if not np.isnan(Tr) else None,
        "n_total": len(df),
        "n_test": n_test,
        "n_forget": n_forget,
        "n_retain": n_retain,
        "n_test_correct": t_correct,
        "n_forget_correct": f_correct,
        "n_retain_correct": r_correct,
    }


def main():
    ap = ArgumentParser()
    ap.add_argument("--config", default="configs/eval_judge.yaml")
    ap.add_argument(
        "--summary_only",
        action="store_true",
        help="Only compute summary from existing results",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = resolve_config_key(
        cfg, "output_dir", "out_dir", required=False, default="out"
    )
    ensure_dir(out_dir)

    scores_path = Path(
        resolve_config_key(
            cfg, "results_path", required=False, default=f"{out_dir}/eval_scores.jsonl"
        )
    )
    false_reasons_path = Path(
        resolve_config_key(
            cfg,
            "false_reasons_path",
            required=False,
            default=f"{out_dir}/eval_false_reasons.jsonl",
        )
    )
    summary_path = Path(
        resolve_config_key(
            cfg,
            "summary_path",
            required=False,
            default=f"{out_dir}/evaluation_summary.json",
        )
    )

    for p in (scores_path, false_reasons_path, summary_path):
        ensure_dir(str(p.parent))

    if args.summary_only:
        if not scores_path.exists():
            print(f"Error: {scores_path} not found")
            return
        print(f"Loading results from {scores_path}...")
        df = pd.DataFrame(read_jsonl(str(scores_path)))
        write_json(str(summary_path), compute_and_print_summary(df))
        print(f"\nSaved to: {summary_path}")
        return

    traces_path = resolve_config_key(cfg, "input_path", "output_path", "output_file")
    traces = read_jsonl(traces_path)
    print(f"Loaded {len(traces)} instances from {traces_path}")

    valid_ids = {item["instance_id"] for item in traces}
    judge_model = resolve_config_key(
        cfg, "model_path", "judge_model_path", "judge_model"
    )
    judge_pipe = load_judge(judge_model)
    done_ids = get_done_ids(scores_path, valid_ids)

    remaining = [item for item in traces if item["instance_id"] not in done_ids]
    print(f"Remaining: {len(remaining)} / {len(traces)}")

    with open(scores_path, "a", encoding="utf-8") as f_scores, open(
        false_reasons_path, "a", encoding="utf-8"
    ) as f_reasons:

        pbar = tqdm(
            total=len(traces),
            initial=len(done_ids),
            desc="Judging",
            unit="sample",
        )
        for item in remaining:
            instance_id = item["instance_id"]
            name = item.get("Name", "unknown")
            print(f"\n[{instance_id}] {name}")

            try:
                used, reason = judge_tool_use(
                    judge_pipe,
                    item["pred_trace"],
                    item["gt_trace"],
                    item["input"],
                )
            except Exception as e:
                print(f"Skip {instance_id}: {e}")
                used, reason = False, ""

            row = {
                "Name": name,
                "instance_id": instance_id,
                "split": item.get("split", "unknown"),
                "used_tool": used,
            }
            f_scores.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_scores.flush()

            if not used and reason:
                f_reasons.write(
                    json.dumps(
                        {
                            "Name": name,
                            "instance_id": instance_id,
                            "split": item.get("split", "unknown"),
                            "reason": reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f_reasons.flush()

            pbar.update(1)
        pbar.close()

    df = pd.DataFrame(read_jsonl(str(scores_path)))
    write_json(str(summary_path), compute_and_print_summary(df))
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()
