import re
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from utils.io_utils import (
    ensure_dir,
    read_jsonl,
    write_json,
    load_config,
    get_done_ids,
    resolve_config_key,
)

_BASELINE_TT = 60.0
_BASELINE_TF = 75.7
_BASELINE_TR = 73.1


def load_judge(model_name: str):
    return OpenAI()


def extract_actions(trace: str) -> str:
    lines = trace.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Action:") or stripped.startswith("Action Input:"):
            result.append(line)
    return "\n".join(result).strip()


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
    "You are evaluating whether an AI assistant correctly called the required tool.\n\n"
    "You will be given the task instruction, the standard answer (Action and Action Input only), "
    "and the assistant's actions (Action and Action Input only).\n\n"
    "Follow these guidelines:\n"
    "1. The assistant must call the same function as the standard answer.\n"
    "2. Calls to getDetails or retrievalDataFromFile as additional steps are acceptable.\n"
    "3. Random calls to unrelated functions are not allowed.\n"
    "4. IMPORTANT: Only judge parameters that are explicitly mentioned in the task instruction. "
    "Parameters that the user did NOT mention (such as pagination, gender, limit, or optional fields) "
    "must be ignored even if they appear in the standard answer.\n\n"
    "Reply with only one word: Yes or No."
)


_SKIP_ACTIONS = {"getDetails", "retrievalDataFromFile"}


def extract_action_names(trace: str) -> list[str]:
    return [m for m in re.findall(r"Action:\s*(\S+)", trace) if m not in _SKIP_ACTIONS]


def judge_tool_use(
    client,
    pred_trace: str,
    gt_trace: str,
    instruction: str,
    model_name: str = "gpt-4o-mini",
) -> tuple[bool, str]:
    gt_trace = strip_duplicate_actions(gt_trace)
    pred_trace_filtered = extract_actions(pred_trace)

    gt_names = extract_action_names(gt_trace)
    pred_names = extract_action_names(pred_trace_filtered)

    if not pred_names:
        print(f"[{instance_id_ctx}] No action -> No")
        return False, "no action"

    if gt_names and not all(a in pred_names for a in gt_names):
        print(
            f"[{instance_id_ctx}] Action mismatch gt={gt_names} pred={pred_names} -> No"
        )
        return False, "action name mismatch"

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"## Task Instruction\n{instruction}\n\n"
                f"## Standard Answer\n{gt_trace}\n\n"
                f"## AI Assistant's Solution\n{pred_trace_filtered}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
    )

    raw_content = response.choices[0].message.content
    if not raw_content:
        print(
            f"[{instance_id_ctx}] Judge: [EMPTY RESPONSE] finish_reason={response.choices[0].finish_reason}"
        )
        correct = any(a in pred_names for a in gt_names)
        return correct, "empty response"

    new_text = response.choices[0].message.content.strip()
    print(f"[{instance_id_ctx}] Judge: {new_text}")

    match = re.search(r"\b(Yes|No)\b", new_text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes", new_text

    correct = any(a in pred_names for a in gt_names)
    return correct, new_text


instance_id_ctx = ""


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
    print(
        f"{'Tt':<8} {Tt:<12.3f} {_BASELINE_TT:<12} {t_correct}/{n_test} ({Tt*100:6.2f}%)"
    )
    print(
        f"{'Tf':<8} {Tf:<12.3f} {_BASELINE_TF:<12} "
        f"{f_correct}/{n_forget} ({Tf*100 if not np.isnan(Tf) else 0:6.2f}%)"
    )
    if np.isnan(Tr):
        print(
            f"{'Tr':<8} {'N/A':<12} {_BASELINE_TR:<12} {r_correct}/{n_retain} (no data)"
        )
    else:
        print(
            f"{'Tr':<8} {Tr:<12.3f} {_BASELINE_TR:<12} "
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
    global instance_id_ctx

    ap = ArgumentParser()
    ap.add_argument("--config", default="configs/eval_judge_openai.yaml")
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
    summary_path = Path(
        resolve_config_key(
            cfg,
            "summary_path",
            required=False,
            default=f"{out_dir}/evaluation_summary.json",
        )
    )

    ensure_dir(str(scores_path.parent))
    ensure_dir(str(summary_path.parent))

    if args.summary_only:
        if not scores_path.exists():
            print(f"Error: {scores_path} not found")
            return
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
    client = load_judge(judge_model)
    done_ids = get_done_ids(scores_path, valid_ids)

    remaining = [item for item in traces if item["instance_id"] not in done_ids]
    print(f"Remaining: {len(remaining)} / {len(traces)}")

    with open(scores_path, "a", encoding="utf-8") as f_scores:
        pbar = tqdm(
            total=len(traces), initial=len(done_ids), desc="Judging", unit="sample"
        )
        for item in remaining:
            instance_id_ctx = item["instance_id"]
            name = item.get("Name", "unknown")

            try:
                used, _ = judge_tool_use(
                    client,
                    item["pred_trace"],
                    item["gt_trace"],
                    item["input"],
                    model_name=judge_model,
                )
            except Exception as e:
                print(f"Skip {instance_id_ctx}: {e}")
                used = False

            row = {
                "Name": name,
                "instance_id": instance_id_ctx,
                "split": item.get("split", "unknown"),
                "used_tool": used,
                "pred_trace": item.get("pred_trace", ""),
            }
            f_scores.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_scores.flush()
            pbar.update(1)
        pbar.close()

    df = pd.DataFrame(read_jsonl(str(scores_path)))
    write_json(str(summary_path), compute_and_print_summary(df))
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()
