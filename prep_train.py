import argparse
import json
import os
import random
from typing import Any, Dict, List


from utils.io_utils import ensure_dir, read_json, write_json, load_config
from utils.trace_utils import build_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prep.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prep_cfg = cfg.get("prep_train", cfg)

    required_keys = [
        "input_path",
        "forget_ratio",
        "split_tools_path",
        "forget_output_path",
        "retain_output_path",
    ]
    missing = [k for k in required_keys if prep_cfg.get(k) is None]
    if missing:
        raise ValueError(f"Missing prep_train config: {', '.join(missing)}")

    for key in ("split_tools_path", "forget_output_path", "retain_output_path"):
        ensure_dir(os.path.dirname(prep_cfg[key]) or ".")
    if prep_cfg.get("flat_instances_path"):
        ensure_dir(os.path.dirname(prep_cfg["flat_instances_path"]) or ".")

    seed = 42
    random.seed(seed)

    tools = read_json(prep_cfg["input_path"])
    if not isinstance(tools, list):
        raise ValueError("Expected a list of tools in input JSON.")

    used_ids: set = set()
    name_max_idx: Dict[str, int] = {}
    all_samples: List[Dict[str, Any]] = []

    for api in tools:
        name = api.get("Name", "")
        instances = api.get("Instances", [])
        for raw_idx, inst in enumerate(instances):
            sample = build_sample(inst, api, name=name, idx=raw_idx)
            if sample is None:
                continue

            base_id = f"{name}_{raw_idx}"
            if base_id not in used_ids:
                instance_id = base_id
                used_ids.add(instance_id)
                name_max_idx[name] = max(name_max_idx.get(name, -1), raw_idx)
            else:
                next_idx = name_max_idx.get(name, raw_idx) + 1
                instance_id = f"{name}_{next_idx}"
                used_ids.add(instance_id)
                name_max_idx[name] = next_idx

            all_samples.append(
                {
                    "Name": name,
                    "instance_id": instance_id,
                    "data": sample,
                }
            )

    all_tool_names = sorted({s["Name"] for s in all_samples})
    print(f"Total samples: {len(all_samples)}, unique tools: {len(all_tool_names)}")

    if prep_cfg.get("flat_instances_path"):
        flat_data = [
            {
                "Name": s["Name"],
                "instance_id": s["instance_id"],
                "process": s["data"][0],
                "trainable": s["data"][1],
            }
            for s in all_samples
        ]
        with open(prep_cfg["flat_instances_path"], "w", encoding="utf-8") as f:
            json.dump(flat_data, f, ensure_ascii=False, indent=4)
        print(
            f"Saved flat: {prep_cfg['flat_instances_path']} ({len(flat_data)} samples)"
        )

    split_tool_names = list(all_tool_names)
    if prep_cfg.get("max_tools"):
        split_tool_names = random.sample(
            split_tool_names, min(len(split_tool_names), prep_cfg["max_tools"])
        )

    random.shuffle(split_tool_names)
    k = max(1, int(len(split_tool_names) * prep_cfg["forget_ratio"]))
    tf_tools = set(split_tool_names[:k])
    tr_tools = set(split_tool_names[k:])

    split_dict = {
        "seed": seed,
        "forget_ratio": prep_cfg["forget_ratio"],
        "num_tools": len(split_tool_names),
        "tf_tools": sorted(tf_tools),
        "tr_tools": sorted(tr_tools),
    }
    write_json(prep_cfg["split_tools_path"], split_dict)

    def to_records(samples):
        return [
            {
                "Name": s["Name"],
                "instance_id": s["instance_id"],
                "process": s["data"][0],
                "trainable": s["data"][1],
            }
            for s in samples
        ]

    forget_data = to_records([s for s in all_samples if s["Name"] in tf_tools])
    retain_data = to_records([s for s in all_samples if s["Name"] in tr_tools])

    for path, data in [
        (prep_cfg["forget_output_path"], forget_data),
        (prep_cfg["retain_output_path"], retain_data),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Forget: {len(tf_tools)} tools / {len(forget_data)} samples")
    print(f"Retain: {len(tr_tools)} tools / {len(retain_data)} samples")


if __name__ == "__main__":
    main()
