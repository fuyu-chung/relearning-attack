#!/usr/bin/env python3
"""fill_and_align_traces.py

針對 judge_results 裡有但 generated_traces 裡沒有的 instance_id，
重新 generate pred_trace，然後把全部結果按 judge_results 的順序輸出。

用法:
    python fill_and_align_traces.py \
        --traces   out/tooldelete/generated_traces_ratio_2_0.jsonl \
        --judges   out/tooldelete/judge_results_ratio_2_0.jsonl \
        --config   configs/eval_generate.yaml \
        --output   out/tooldelete/generated_traces_ratio_2_0_aligned.jsonl
"""
import json
import torch
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from utils.io_utils import read_jsonl, load_config, load_model, resolve_config_key
from utils.trace_utils import (
    get_tool_names,
    serialize_golden,
    serialize_golden_from_steps,
    load_forget_instances,
    PREFIX_TRAIN,
    FORMAT_INSTRUCTIONS_TRAIN,
    GET_DETAILS_DESCRIPTION,
)


def build_prompt(user_input: str, nl_doc: str, tool_names: str) -> str:
    all_tool_names = f"getDetails, {tool_names}" if tool_names else "getDetails"
    fmt = FORMAT_INSTRUCTIONS_TRAIN.replace("{tool_names}", all_tool_names)
    return (
        f"{PREFIX_TRAIN}\n\n"
        f"{GET_DETAILS_DESCRIPTION}\n"
        f"{nl_doc}\n\n"
        f"{fmt}\n\n"
        f"Begin!\n\n"
        f"Question: {user_input}\n"
        "Thought:"
    )


def generate_trace(tokenizer, model, user_input: str, nl_doc: str, tool_names: str) -> str:
    prompt = build_prompt(user_input, nl_doc, tool_names)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def build_instance_lookup(cfg: dict) -> dict:
    """Build instance_id -> {nl_doc, tool_names, input, gt_trace} from eval + train splits."""
    from utils.io_utils import read_json

    lookup = {}

    # eval split
    eval_tools = read_json(resolve_config_key(cfg, "eval_tools_path"))
    for t in eval_tools:
        name = t.get("Name", "")
        nl_doc = t.get("NLDocumentation", "")
        tool_names = get_tool_names(nl_doc)
        for i, (instruction, golden) in enumerate(
            zip(t.get("Instructions", []), t.get("Golden_Answers", []))
        ):
            gt_trace = serialize_golden(golden)
            if not instruction or not gt_trace:
                continue
            lookup[f"{name}_{i}"] = {
                "Name": name,
                "nl_doc": nl_doc,
                "tool_names": tool_names,
                "input": instruction.strip(),
                "gt_trace": gt_trace,
                "split": "test",
            }

    # forget + retain splits
    train_tools_path = resolve_config_key(cfg, "train_tools_path")
    for split, path_key in [("forget", "forget_data_path"), ("retain", "retain_data_path")]:
        data_path = resolve_config_key(cfg, path_key)
        instances = load_forget_instances(data_path, train_tools_path)
        for inst in instances:
            gt_trace = inst.get("gt_trace", "")
            if not inst["question"] or not gt_trace:
                continue
            lookup[inst["instance_id"]] = {
                "Name": inst["Name"],
                "nl_doc": inst["nl_doc"],
                "tool_names": inst["tool_names"],
                "input": inst["question"],
                "gt_trace": gt_trace,
                "split": split,
            }

    return lookup


def main():
    ap = ArgumentParser()
    ap.add_argument("--traces",  required=True, help="existing generated_traces.jsonl")
    ap.add_argument("--judges",  required=True, help="judge_results.jsonl")
    ap.add_argument("--config",  default="configs/eval_generate.yaml")
    ap.add_argument("--output",  required=True, help="output path for aligned traces")
    args = ap.parse_args()

    judges = read_jsonl(args.judges)
    traces = read_jsonl(args.traces)

    judge_ids = [r["instance_id"] for r in judges]
    trace_map = {r["instance_id"]: r for r in traces}

    missing_ids = [iid for iid in judge_ids if iid not in trace_map]
    print(f"Total judge ids: {len(judge_ids)}")
    print(f"Existing traces: {len(trace_map)}")
    print(f"Missing: {len(missing_ids)} -> {missing_ids}")

    if missing_ids:
        cfg = load_config(args.config)
        model_path = resolve_config_key(cfg, "model_path", "adapter_model_path", "base_model_path")
        base_model = cfg.get("base_model_path")
        offload_dir = cfg.get("offload_dir")

        lookup = build_instance_lookup(cfg)
        tokenizer, model = load_model(model_path, base_model, offload_dir)

        for iid in tqdm(missing_ids, desc="Generating missing traces"):
            inst = lookup.get(iid)
            if inst is None:
                print(f"  Warning: {iid} not found in any split, skipping")
                trace_map[iid] = {
                    "Name": iid.rsplit("_", 1)[0],
                    "instance_id": iid,
                    "split": "unknown",
                    "input": "",
                    "gt_trace": "",
                    "pred_trace": "",
                }
                continue

            try:
                pred_trace = generate_trace(
                    tokenizer, model,
                    inst["input"], inst["nl_doc"], inst["tool_names"],
                )
            except Exception as e:
                print(f"  Skip {iid}: {e}")
                pred_trace = ""

            trace_map[iid] = {
                "Name": inst["Name"],
                "instance_id": iid,
                "split": inst["split"],
                "input": inst["input"],
                "gt_trace": inst["gt_trace"],
                "pred_trace": pred_trace,
            }
            print(f"  Generated: {iid}")

        del model
        torch.cuda.empty_cache()

    # write aligned output in judge order
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for iid in judge_ids:
            row = trace_map.get(iid)
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nAligned traces saved: {out_path} ({len(judge_ids)} rows)")


if __name__ == "__main__":
    main()