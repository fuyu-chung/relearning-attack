import json
import yaml
from argparse import ArgumentParser


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_gt_trace(gt_trace: str) -> str:
    """移除 Observation，並去除重複的 Action/Action Input"""
    lines = gt_trace.split("\n")

    # 移除 Observation
    result = []
    skip = False
    for line in lines:
        if line.startswith("Observation:"):
            skip = True
        elif (
            line.startswith("Thought:")
            or line.startswith("Action:")
            or line.startswith("Action Input:")
            or line.startswith("Final Answer:")
            or line == ""
        ):
            skip = False
        if not skip:
            result.append(line)

    # 去除重複的 Action/Action Input 組合
    seen = set()
    deduped = []
    i = 0
    while i < len(result):
        line = result[i]
        if line.startswith("Action:") and i + 1 < len(result) and result[i + 1].startswith("Action Input:"):
            key = (line, result[i + 1])
            if key not in seen:
                seen.add(key)
                deduped.append(line)
                deduped.append(result[i + 1])
            i += 2
        else:
            deduped.append(line)
            i += 1

    return "\n".join(deduped).strip()


def main():
    ap = ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--input", help="Override input file path")
    ap.add_argument("--output", help="Override output file path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    input_path = args.input or cfg["output_file"]
    output_path = args.output or input_path.replace(".jsonl", "_clean.jsonl")

    with open(input_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(rows)} instances from {input_path}")

    for row in rows:
        row["gt_trace"] = clean_gt_trace(row["gt_trace"])

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()