from argparse import ArgumentParser

from utils.io_utils import read_json, write_jsonl, load_config, resolve_config_key
from utils.trace_utils import get_tool_names, serialize_golden, build_sft_row


def flatten_eval(tools: list) -> list:
    rows = []
    for tool in tools:
        name = tool.get("Name", "")
        instructions = tool.get("Instructions", [])
        golden_answers = tool.get("Golden_Answers", [])

        for i, (instruction, golden) in enumerate(zip(instructions, golden_answers)):
            gt_trace = serialize_golden(golden, include_observation=True)
            if not instruction or not gt_trace:
                print(f"Skipping {name}_{i} due to empty instruction or gt_trace")
                continue

            rows.append(
                build_sft_row(
                    name=name,
                    instance_id=f"{name}_{i}",
                    split="test",
                    user_input=instruction.strip(),
                    assistant_output=gt_trace,
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

    input_path = resolve_config_key(eval_cfg, "input_path")
    output_path = resolve_config_key(eval_cfg, "output_path")

    tools = read_json(input_path)
    if not isinstance(tools, list):
        raise ValueError("Expected a list of tools in the input JSON.")

    rows = flatten_eval(tools)
    write_jsonl(output_path, rows)
    print(f"Saved {output_path}: {len(rows)} instances from {len(tools)} tools")


if __name__ == "__main__":
    main()
