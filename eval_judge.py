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
    tokenizer.padding_side = "left"

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
        batch_size=8,
    )
    return judge_pipe


def strip_observation(trace: str) -> str:
    cleaned = re.sub(
        r"\n?\s*Observation:.*?(?=\n\s*(Action|Thought|Final Answer)|$)",
        "",
        trace,
        flags=re.DOTALL,
    )
    return cleaned.strip()


def strip_duplicate_actions(trace: str) -> str:
    """移除 gt_trace 裡連續重複的 Action + Action Input 區塊"""
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


def judge_tool_use(
    judge_pipe, pred_trace: str, gt_trace: str, instruction: str
) -> bool:
    gt_trace = strip_duplicate_actions(gt_trace)
    pred_trace = strip_observation(pred_trace)

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
        correct = match.group(1).lower() == "yes"
        return correct, new_text

    gt_actions = re.findall(r"Action:\s*(\S+)", gt_trace)
    correct = any(action in pred_trace for action in gt_actions)
    return correct, new_text







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
    out_dir = cfg.get("output_dir", cfg.get("out_dir", "out"))
    ensure_dir(out_dir)

    judge_output_dir = cfg.get("judge_output_dir", out_dir)
    scores_filename = cfg.get("judge_results_file", "eval_scores.jsonl")
    false_reasons_filename = cfg.get("judge_false_reasons_file", "eval_false_reasons.jsonl")
    summary_filename = cfg.get("judge_summary_file", "evaluation_summary.json")

    scores_path = Path(cfg.get("results_path", str(Path(judge_output_dir) / scores_filename)))
    false_reasons_path = Path(
        cfg.get(
            "false_reasons_path",
            str(Path(judge_output_dir) / false_reasons_filename),
        )
    )
    summary_path = Path(cfg.get("summary_path", str(Path(judge_output_dir) / summary_filename)))

    ensure_dir(str(scores_path.parent))
    ensure_dir(str(false_reasons_path.parent))
    ensure_dir(str(summary_path.parent))

    if args.summary_only:
        if not scores_path.exists():
            print(f"Error: {scores_path} not found")
            return
        print(f"Loading results from {scores_path}...")
        all_scores = read_jsonl(str(scores_path))
        df = pd.DataFrame(all_scores)
    else:
        traces_input_path = cfg.get(
            "input_path", cfg.get("output_path", cfg.get("output_file"))
        )
        if not traces_input_path:
            raise ValueError(
                "Need input_path (or output_path / legacy output_file) for judge input"
            )
        traces = read_jsonl(traces_input_path)
        print(f"Loaded {len(traces)} instances from {traces_input_path}")

        valid_ids = {item["instance_id"] for item in traces}

        judge_model = cfg.get("judge_model_path", cfg.get("judge_model"))
        if not judge_model:
            raise ValueError("Need judge_model_path (or legacy judge_model)")
        judge_pipe = load_judge(judge_model)

        done_ids = set()
        if scores_path.exists():
            done = read_jsonl(str(scores_path))
            raw_done_ids = {r["instance_id"] for r in done if "instance_id" in r}
            done_ids = raw_done_ids & valid_ids
            print(f"Resuming: {len(done_ids)} already done")
            ignored = len(raw_done_ids) - len(done_ids)
            if ignored > 0:
                print(f"Ignored {ignored} done ids not in current trace set")

        total = len(traces)
        remaining = [item for item in traces if item["instance_id"] not in done_ids]
        print(f"Remaining: {len(remaining)} / {total}")

        print(f"\nStart judging ({total} samples)...")

        with open(scores_path, "a", encoding="utf-8") as f, open(
            false_reasons_path, "a", encoding="utf-8"
        ) as fr:
            pbar = tqdm(
                total=total,
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
                    row = {
                        "Name": name,
                        "instance_id": instance_id,
                        "split": item.get("split", "unknown"),
                        "used_tool": used,
                    }
                    if not used:
                        fr.write(
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
                        fr.flush()
                except Exception as e:
                    print(f"Skip {instance_id}: {e}")
                    row = {
                        "Name": name,
                        "instance_id": instance_id,
                        "split": item.get("split", "unknown"),
                        "used_tool": False,
                    }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                pbar.update(1)

            pbar.close()

        all_scores = read_jsonl(str(scores_path))
        df = pd.DataFrame(all_scores)

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
    print(f"{'Tf':<8} {Tf:<12.3f} {'75.7':<12} {f_correct}/{n_forget} ({Tf*100 if not np.isnan(Tf) else 0:6.2f}%)")
    if np.isnan(Tr):
        print(f"{'Tr':<8} {'N/A':<12} {'73.1':<12} {r_correct}/{n_retain} (no data)")
    else:
        print(f"{'Tr':<8} {Tr:<12.3f} {'73.1':<12} {r_correct}/{n_retain} ({Tr*100:6.2f}%)")
    print("=" * 70)

    write_json(
        str(summary_path),
        {
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
        },
    )

    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()
