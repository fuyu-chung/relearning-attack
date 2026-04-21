import argparse
import os
import random
from typing import Any, Dict, List, Tuple

from utils.io_utils import (
    ensure_dir,
    read_json,
    write_json,
    write_jsonl,
    load_config,
    safe_str,
)


def parse_intermediate_steps(steps: Any) -> List[Tuple[str, str, str, str]]:
    parsed = []
    if not isinstance(steps, list):
        return parsed
    for step in steps:
        if not isinstance(step, list) or len(step) < 2:
            continue
        action_part, obs_part = step[0], step[1]
        triplet = None
        if isinstance(action_part, list):
            if action_part and isinstance(action_part[0], list):
                triplet = action_part[0]
            elif len(action_part) >= 2 and isinstance(action_part[0], str):
                triplet = action_part
        if not isinstance(triplet, list) or len(triplet) < 2:
            continue
        action = safe_str(triplet[0]).strip()
        action_input = safe_str(triplet[1]).strip()
        thought = safe_str(triplet[2]).strip() if len(triplet) >= 3 else ""
        thought = thought.split("\nAction:")[0].strip() if thought else ""
        observation = safe_str(obs_part).strip()
        parsed.append((thought, action, action_input, observation))
    return parsed


def serialize_trace(instance: Dict[str, Any]) -> str:
    lines = []
    for thought, action, action_input, obs in parse_intermediate_steps(
        instance.get("intermediate_steps", [])
    ):
        if thought:
            lines.append(f"Thought: {thought}")
        if action:
            lines.append(f"Action: {action}")
        if action_input:
            lines.append(f"Action Input: {action_input}")
        if obs:
            lines.append(f"Observation: {obs}")
        lines.append("")
    final = safe_str(instance.get("output", "")).strip()
    if final:
        lines.append(f"Final Answer: {final}")
    return "\n".join(lines).strip()


def flatten_tools(
    tools: List[Dict[str, Any]], verbose: bool = False
) -> List[Dict[str, Any]]:
    flat = []
    used_ids = set()
    name_max_idx = {}

    for t in tools:
        name = t.get("Name", "")
        instances = t.get("Instances", [])

        if not isinstance(instances, list):
            continue

        for i, inst in enumerate(instances):
            if not isinstance(inst, dict):
                continue

            user = safe_str(inst.get("input", "")).strip()
            trace = serialize_trace(inst).strip()
            if not user or not trace:
                continue

            base_id = f"{name}_{i}"
            if base_id not in used_ids:
                instance_id = base_id
                used_ids.add(instance_id)
                name_max_idx[name] = max(name_max_idx.get(name, -1), i)
            else:
                next_idx = name_max_idx.get(name, i) + 1
                instance_id = f"{name}_{next_idx}"
                used_ids.add(instance_id)
                name_max_idx[name] = next_idx
                if verbose:
                    print(f"Duplicate: {base_id} -> {instance_id}")

            flat.append(
                {
                    "Name": name,
                    "instance_id": instance_id,
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": trace},
                    ],
                }
            )

    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prep.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    prep_cfg = cfg.get("prep_train") if isinstance(cfg, dict) else None
    if not isinstance(prep_cfg, dict):
        raise ValueError("Missing prep_train section in config")

    required_keys = [
        "input_path",
        "forget_ratio",
        "flat_instances_path",
        "split_tools_path",
        "forget_output_path",
        "retain_output_path",
    ]
    missing = [k for k in required_keys if prep_cfg.get(k) is None]
    if missing:
        raise ValueError(f"Missing prep_train config: {', '.join(missing)}")

    in_json = prep_cfg["input_path"]
    forget_ratio = prep_cfg["forget_ratio"]
    max_train_tools = prep_cfg.get("max_tools")
    seed = 42
    flat_instances_path = prep_cfg["flat_instances_path"]
    split_tools_path = prep_cfg["split_tools_path"]
    forget_output_path = prep_cfg["forget_output_path"]
    retain_output_path = prep_cfg["retain_output_path"]

    for p in [
        flat_instances_path,
        split_tools_path,
        forget_output_path,
        retain_output_path,
    ]:
        ensure_dir(os.path.dirname(p) or ".")

    random.seed(seed)

    tools = read_json(in_json)
    if not isinstance(tools, list):
        raise ValueError("Expected a list of tools in the input JSON.")

    # 1. train_flat_new.jsonl 一定是全部攤平的資料
    flat = flatten_tools(tools, verbose=True)
    all_tool_names = sorted({r["Name"] for r in flat if r["Name"]})
    write_jsonl(flat_instances_path, flat)
    print(f"Saved {flat_instances_path}: {len(all_tool_names)}/{len(flat)} tools/instances")

    # 2. max_tools 只影響 split/forget/retain 三個檔案
    if max_train_tools:
        random.seed(seed)
        if len(all_tool_names) < max_train_tools:
            print(f"Warning: only {len(all_tool_names)} unique tools, less than max_tools={max_train_tools}")
            selected_tools = all_tool_names
        else:
            selected_tools = random.sample(all_tool_names, max_train_tools)
    else:
        selected_tools = all_tool_names

    split_tool_names = list(selected_tools)
    random.shuffle(split_tool_names)
    k = max(1, int(len(split_tool_names) * forget_ratio))
    tf_tools = set(split_tool_names[:k])
    tr_tools = set(split_tool_names[k:])

    split_dict = {
        "seed": seed,
        "forget_ratio": forget_ratio,
        "num_tools": len(split_tool_names),
        "tf_tools": sorted(tf_tools),
        "tr_tools": sorted(tr_tools),
    }
    write_json(split_tools_path, split_dict)

    # forget/retain 依 split_tools 分配
    forget_rows = [r for r in flat if r["Name"] in split_dict["tf_tools"]]
    retain_rows = [r for r in flat if r["Name"] in split_dict["tr_tools"]]

    write_jsonl(forget_output_path, forget_rows)
    write_jsonl(retain_output_path, retain_rows)

    print(f"Forget tools/instances: {len(split_dict['tf_tools'])}/{len(forget_rows)} | Retain tools/instances: {len(split_dict['tr_tools'])}/{len(retain_rows)}")


if __name__ == "__main__":
    main()
