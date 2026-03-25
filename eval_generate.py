import json
import re
import yaml
import torch
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.io_utils import ensure_dir, read_json, read_jsonl, write_json


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        local_files_only=True,
        legacy=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def get_tool_names(nl_doc: str) -> str:
    _SKIP = {"Parameters", "Output", "Structure", "Format"}
    names = [
        m
        for m in re.findall(r"^([A-Za-z]\w*):", nl_doc, re.MULTILINE)
        if m not in _SKIP
    ]
    return ", ".join(names) if names else ""


def safe_str(x) -> str:
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
        action = safe_str(step.get("Action", "")).strip()
        action_input = safe_str(step.get("Action_Input", "")).strip()
        thought = safe_str(step.get("Thought", "")).strip()
        if thought:
            lines.append(f"Thought: {thought}")
        if action:
            lines.append(f"Action: {action}")
        if action_input:
            lines.append(f"Action Input: {action_input}")
        lines.append("")
    if actions:
        final = safe_str(actions[-1].get("Final_Answer", "")).strip()
        if final:
            lines.append(f"Final Answer: {final}")
    return "\n".join(lines).strip()


def parse_intermediate_steps(steps) -> list:
    parsed = []
    if not isinstance(steps, list):
        return parsed
    for step in steps:
        if not isinstance(step, list) or len(step) < 1:
            continue
        action_part = step[0]
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
        raw_thought = safe_str(triplet[2]).strip() if len(triplet) >= 3 else ""
        thought = raw_thought.split("\nAction:")[0].strip()
        parsed.append((thought, action, action_input))
    return parsed


def serialize_trace(instance: dict) -> str:
    lines = []
    for thought, action, action_input in parse_intermediate_steps(
        instance.get("intermediate_steps", [])
    ):
        if thought:
            lines.append(f"Thought: {thought}")
        if action:
            lines.append(f"Action: {action}")
        if action_input:
            lines.append(f"Action Input: {action_input}")
        lines.append("")
    final = safe_str(instance.get("output", "")).strip()
    if final:
        lines.append(f"Final Answer: {final}")
    return "\n".join(lines).strip()


def load_data(cfg: dict) -> list:
    split = read_json(str(Path(cfg["out_dir"]) / "split_tools.json"))
    tf_tools = set(split["tf_tools"])
    tr_tools = set(split["tr_tools"])

    eval_data = []
    forget_data = []
    retain_data = []

    eval_tools = read_json(cfg["eval_raw"])
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
            eval_data.append(
                {
                    "Name": name,
                    "instance_id": f"{name}_{i}",
                    "nl_doc": nl_doc,
                    "tool_names": tool_names,
                    "user_input": instruction.strip(),
                    "gt_trace": gt_trace,
                    "_split": "test",
                }
            )

    from pathlib import Path as _Path

    forget_sft = read_jsonl(cfg["forget_sft"])
    retain_sft = read_jsonl(cfg["retain_sft"])

    nl_map = {}
    train_tools = read_json(cfg["in_json"])
    for t in train_tools:
        name = t.get("Name", "")
        nl_doc = t.get("NLDocumentation", "")
        nl_map[name] = (nl_doc, get_tool_names(nl_doc))

    for r in forget_sft:
        name = r["Name"]
        iid = r["instance_id"]
        user = r.get("messages", [{}])[0].get("content", "")
        nl_doc, tool_names_str = nl_map.get(name, ("", ""))
        forget_data.append(
            {
                "Name": name,
                "instance_id": iid,
                "nl_doc": nl_doc,
                "tool_names": tool_names_str,
                "user_input": user,
                "gt_trace": (
                    r.get("messages", [{}, {}])[1].get("content", "")
                    if len(r.get("messages", [])) > 1
                    else ""
                ),
                "_split": "forget",
            }
        )

    for r in retain_sft:
        name = r["Name"]
        iid = r["instance_id"]
        user = r.get("messages", [{}])[0].get("content", "")
        nl_doc, tool_names_str = nl_map.get(name, ("", ""))
        retain_data.append(
            {
                "Name": name,
                "instance_id": iid,
                "nl_doc": nl_doc,
                "tool_names": tool_names_str,
                "user_input": user,
                "gt_trace": (
                    r.get("messages", [{}, {}])[1].get("content", "")
                    if len(r.get("messages", [])) > 1
                    else ""
                ),
                "_split": "retain",
            }
        )

    all_data = eval_data + forget_data + retain_data
    print(
        f"  eval={len(eval_data)}, forget={len(forget_data)}, retain={len(retain_data)}"
    )
    print(f"Total samples: {len(all_data)}")

    return all_data


def build_prompt(user_input: str, nl_doc: str, tool_names: str) -> str:
    return f"""Your task is to answer the user's question using available tools. The user cannot see or use the tools themselves, nor can they know the process of your tool usage. Provide all necessary information in the "Final Answer" field. Do not make up any information. If required parameters are missing, use the "getDetails" tool to ask the user for them.
You have access to the following tools:

{nl_doc}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action, must be in JSON format. All of the action input must be realistic and from the user.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: Summarize the information gathered and the reasoning behind your final answer.
Final Answer: Provide a user-friendly and detailed answer to the original input question that summarizes all relevant information from the Thought/Action/Action Input/Observation sequences.

Begin!

Question: {user_input}
Thought:"""


def generate_trace(
    tokenizer, model, user_input: str, nl_doc: str, tool_names: str
) -> str:
    prompt = build_prompt(user_input, nl_doc, tool_names)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()


def main():
    ap = ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    ensure_dir(cfg["out_dir"])
    tokenizer, model = load_model(cfg["toolalpaca_model"])
    eval_data = load_data(cfg)

    output_path = Path(cfg["output_file"])
    done_ids = set()
    if output_path.exists():
        done = read_jsonl(str(output_path))
        done_ids = {r["instance_id"] for r in done}
        print(f"Resuming: {len(done_ids)} already done")

    print(f"\nStart generating ({len(eval_data)} samples)...")

    total = len(eval_data)
    remaining = [item for item in eval_data if item["instance_id"] not in done_ids]
    print(f"Remaining: {len(remaining)} / {total}")

    with open(output_path, "a", encoding="utf-8") as f:
        for item in tqdm(
            remaining, desc=f"Generating traces (done={len(done_ids)}/{total})"
        ):
            instance_id = item["instance_id"]

            try:
                pred_trace = generate_trace(
                    tokenizer,
                    model,
                    item["user_input"],
                    item["nl_doc"],
                    item["tool_names"],
                )
                print(f"\n[{instance_id}] {item['Name']}")
                print(f"Q: {item['user_input']}")
                print(f"Pred: {pred_trace}")
                print("─" * 60)
                row = {
                    "Name": item["Name"],
                    "instance_id": instance_id,
                    "split": item["_split"],
                    "input": item["user_input"],
                    "gt_trace": item["gt_trace"],
                    "pred_trace": pred_trace,
                }
            except Exception as e:
                print(f"Skip {instance_id}: {e}")
                row = {
                    "Name": item["Name"],
                    "instance_id": instance_id,
                    "split": item["_split"],
                    "input": item["user_input"],
                    "gt_trace": item["gt_trace"],
                    "pred_trace": "",
                }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\nSaved to: {cfg['output_file']}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
