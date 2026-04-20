import json
import re
import yaml
from argparse import ArgumentParser
from typing import Any
from utils.io_utils import read_json, write_jsonl


def load_config(path: str) -> dict:
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


def serialize_golden(actions: list) -> str:
    lines = []
    for step in actions:
        thought = safe_str(step.get("Thought", "")).strip()
        action = safe_str(step.get("Action", "")).strip()
        action_input = safe_str(step.get("Action_Input", "")).strip()
        observation = safe_str(step.get("Observation", "")).strip()

        if thought:
            lines.append(f"Thought: {thought}")
        if action:
            lines.append(f"Action: {action}")
        if action_input:
            lines.append(f"Action Input: {action_input}")
        if observation:
            lines.append(f"Observation: {observation}")
        lines.append("")

    if actions:
        final = safe_str(actions[-1].get("Final_Answer", "")).strip()
        if final:
            lines.append(f"Final Answer: {final}")

    return "\n".join(lines).strip()


def get_tool_names(nl_doc: str) -> str:
    """從 NLDocumentation 抽出所有函式名稱，組成逗號分隔的字串"""
    names = re.findall(r"^(\w+):", nl_doc, re.MULTILINE)
    return ", ".join(names) if names else ""


def build_eval_row(
    name: str,
    instance_id: str,
    nl_doc: str,
    tool_names: str,
    instruction: str,
    gt_trace: str,
) -> dict:
    return {
        "Name": name,
        "instance_id": instance_id,
        "nl_doc": nl_doc,
        "tool_names": tool_names,
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": gt_trace},
        ],
    }


def flatten_eval(tools: list) -> list:
    rows = []
    for tool in tools:
        name = tool.get("Name", "")
        instructions = tool.get("Instructions", [])
        golden_answers = tool.get("Golden_Answers", [])
        nl_doc = tool.get("NLDocumentation", "")
        tool_names = get_tool_names(nl_doc)

        for i, (instruction, golden) in enumerate(zip(instructions, golden_answers)):
            gt_trace = serialize_golden(golden)
            if not instruction or not gt_trace:
                print(f"Skipping {name}_{i} due to empty instruction or gt_trace")
                continue

            rows.append(
                build_eval_row(
                    name,
                    f"{name}_{i}",
                    nl_doc,
                    tool_names,
                    instruction.strip(),
                    gt_trace,
                )
            )

    return rows


def main():
    ap = ArgumentParser()
    ap.add_argument("--config", default="configs/prep.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    eval_cfg = cfg.get("prep_eval") if isinstance(cfg, dict) else None
    if not isinstance(eval_cfg, dict):
        raise ValueError("Missing prep_eval section in config")

    required_keys = ["input_path", "output_path"]
    missing = [k for k in required_keys if eval_cfg.get(k) is None]
    if missing:
        raise ValueError(f"Missing prep_eval config: {', '.join(missing)}")

    eval_raw = eval_cfg["input_path"]
    eval_file = eval_cfg["output_path"]

    tools = read_json(eval_raw)
    if not isinstance(tools, list):
        raise ValueError("Expected a list of tools in the input JSON.")

    rows = flatten_eval(tools)
    write_jsonl(eval_file, rows)

    print(f"Saved {eval_file}: {len(rows)} instances from {len(tools)} tools")


if __name__ == "__main__":
    main()
