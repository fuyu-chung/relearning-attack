import json
import torch
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from utils.io_utils import (
    ensure_dir,
    read_json,
    load_config,
    load_model,
    get_done_ids,
    resolve_config_key,
)
from utils.trace_utils import (
    get_tool_names,
    serialize_golden,
    load_forget_instances,
    generate_trace,
)


def load_eval_split(eval_tools_path: str) -> list:
    eval_tools = read_json(eval_tools_path)
    rows = []
    for t in eval_tools:
        name = t.get("Name", "")
        nl_doc = t.get("NLDocumentation", "")
        tool_names = get_tool_names(nl_doc)
        instructions = t.get("Instructions", [])
        golden_answers = t.get("Golden_Answers", [])
        for i, (instruction, golden) in enumerate(zip(instructions, golden_answers)):
            gt_trace = serialize_golden(golden)
            if not instruction or not gt_trace:
                continue
            rows.append(
                {
                    "Name": name,
                    "instance_id": f"{name}_{i}",
                    "nl_doc": nl_doc,
                    "tool_names": tool_names,
                    "input": instruction.strip(),
                    "gt_trace": gt_trace,
                    "split": "test",
                }
            )
    return rows


def instances_to_eval_rows(instances: list, split: str) -> list:
    rows = []
    for inst in instances:
        gt_trace = inst.get("gt_trace", "")
        if not inst["question"] or not gt_trace:
            print(f"Drop {split}/{inst['instance_id']}: empty input or gt_trace")
            continue
        rows.append(
            {
                "Name": inst["Name"],
                "instance_id": inst["instance_id"],
                "nl_doc": inst["nl_doc"],
                "tool_names": inst["tool_names"],
                "input": inst["question"],
                "gt_trace": gt_trace,
                "split": split,
            }
        )
    return rows


def load_data(cfg: dict) -> list:
    eval_tools_path = resolve_config_key(cfg, "eval_tools_path")
    train_tools_path = resolve_config_key(cfg, "train_tools_path")
    forget_data_path = resolve_config_key(cfg, "forget_data_path")
    retain_data_path = resolve_config_key(cfg, "retain_data_path")

    eval_data = load_eval_split(eval_tools_path)

    forget_instances = load_forget_instances(forget_data_path, train_tools_path)
    retain_instances = load_forget_instances(retain_data_path, train_tools_path)
    forget_data = instances_to_eval_rows(forget_instances, "forget")
    retain_data = instances_to_eval_rows(retain_instances, "retain")

    all_data = eval_data + forget_data + retain_data
    print(
        f"  eval={len(eval_data)}, forget={len(forget_data)}, retain={len(retain_data)}"
    )
    print(f"Total samples: {len(all_data)}")
    return all_data


def main():
    ap = ArgumentParser()
    ap.add_argument("--config", default="configs/eval_generate.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    model_path = resolve_config_key(
        cfg, "model_path", "adapter_model_path", "base_model_path"
    )
    base_model = cfg.get("base_model_path")
    offload_dir = cfg.get("offload_dir")
    output_path = Path(resolve_config_key(cfg, "output_path"))

    ensure_dir(str(output_path.parent) if str(output_path.parent) else ".")
    tokenizer, model = load_model(model_path, base_model, offload_dir)

    eval_data = load_data(cfg)
    valid_ids = {item["instance_id"] for item in eval_data}
    done_ids = get_done_ids(output_path, valid_ids)

    remaining = [item for item in eval_data if item["instance_id"] not in done_ids]
    print(f"Remaining: {len(remaining)} / {len(eval_data)}")

    with open(output_path, "a", encoding="utf-8") as f:
        pbar = tqdm(
            total=len(eval_data),
            initial=len(done_ids),
            desc="Generating traces",
            unit="sample",
        )
        for item in remaining:
            instance_id = item["instance_id"]
            try:
                pred_trace = generate_trace(
                    tokenizer,
                    model,
                    item["input"],
                    item["nl_doc"],
                    item["tool_names"],
                )
            except Exception as e:
                print(f"Skip {instance_id}: {e}")
                pred_trace = ""

            if not pred_trace:
                print(f"Skip {instance_id}: empty pred_trace, will retry next run")
                pbar.update(1)
                continue

            row = {
                "Name": item["Name"],
                "instance_id": instance_id,
                "split": item["split"],
                "input": item["input"],
                "gt_trace": item["gt_trace"],
                "pred_trace": pred_trace,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            pbar.update(1)
        pbar.close()

    print(f"\nSaved to: {output_path}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
