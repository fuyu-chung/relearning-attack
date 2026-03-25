import argparse
import json
import os
import random
import yaml
from typing import Any, Dict, List, Tuple

from utils.io_utils import ensure_dir, read_json, write_json, write_jsonl


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


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
    name_max_idx = {}  # 記錄每個工具目前最大的 index

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
                # 接在目前最大 index 後面
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

    # 把同工具的 instance 排在一起，按 index 數字排序，工具間保持原始順序
    import re

    tool_order = []
    tool_groups = {}
    for r in flat:
        name = r["Name"]
        if name not in tool_groups:
            tool_order.append(name)
            tool_groups[name] = []
        tool_groups[name].append(r)

    def get_index(r):
        m = re.search(r"_(\d+)$", r["instance_id"])
        return int(m.group(1)) if m else 0

    flat = [r for name in tool_order for r in sorted(tool_groups[name], key=get_index)]

    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    forget_ratio = cfg["forget_ratio"]

    ensure_dir(cfg["out_dir"])
    tools = read_json(cfg["in_json"])
    if not isinstance(tools, list):
        raise ValueError("Expected a list of tools in the input JSON.")

    # 先存全部工具的 instances（不 print 重複）
    flat_all = flatten_tools(tools, verbose=False)
    all_tool_names = sorted({r["Name"] for r in flat_all if r["Name"]})
    write_jsonl(os.path.join(cfg["out_dir"], "flat_instances.jsonl"), flat_all)
    print(
        f"Saved flat_instances.jsonl: {len(all_tool_names)}/{len(flat_all)} tools/instances"
    )

    # 限制 train 工具數
    max_train_tools = cfg.get("max_train_tools", None)
    if max_train_tools:
        tools = tools[:max_train_tools]
        print(f"Limiting to {max_train_tools} tools")

    flat = flatten_tools(tools, verbose=True)

    tool_names = sorted({r["Name"] for r in flat if r["Name"]})
    rnd = random.Random(42)
    rnd.shuffle(tool_names)
    k = max(1, int(len(tool_names) * forget_ratio))
    tf_tools = set(tool_names[:k])
    tr_tools = set(tool_names[k:])

    write_json(
        os.path.join(cfg["out_dir"], "split_tools.json"),
        {
            "seed": 42,
            "forget_ratio": forget_ratio,
            "num_tools": len(tool_names),
            "tf_tools": sorted(tf_tools),
            "tr_tools": sorted(tr_tools),
        },
    )

    forget_rows = [r for r in flat if r["Name"] in tf_tools]
    retain_rows = [r for r in flat if r["Name"] in tr_tools]

    write_jsonl(os.path.join(cfg["out_dir"], "forget_sft.jsonl"), forget_rows)
    write_jsonl(os.path.join(cfg["out_dir"], "retain_sft.jsonl"), retain_rows)

    print(f"Total flattened tools/instances: {len(tool_names)}/{len(flat)}")
    print(f"Forget tools/instances: {len(tf_tools)}/{len(forget_rows)}")
    print(f"Retain tools/instances: {len(tr_tools)}/{len(retain_rows)}")


if __name__ == "__main__":
    main()
