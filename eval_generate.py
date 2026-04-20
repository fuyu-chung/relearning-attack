import json
import re
import torch
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from utils.io_utils import (
    ensure_dir,
    read_json,
    read_jsonl,
    load_config,
    load_model,
    safe_str,
)


def get_tool_names(nl_doc: str) -> str:
    _SKIP = {"Parameters", "Output", "Structure", "Format"}
    names = [
        m
        for m in re.findall(r"^([A-Za-z]\w*):", nl_doc, re.MULTILINE)
        if m not in _SKIP
    ]
    return ", ".join(names) if names else ""


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


def build_output_row(
    name: str,
    instance_id: str,
    split: str,
    user_input: str,
    gt_trace: str,
    pred_trace: str,
) -> dict:
    return {
        "Name": name,
        "instance_id": instance_id,
        "split": split,
        "input": user_input,
        "gt_trace": gt_trace,
        "pred_trace": pred_trace,
    }


def load_data(cfg: dict) -> list:
    eval_data = []
    forget_data = []
    retain_data = []

    eval_raw_path = cfg.get("eval_tools_path")
    train_tools_path = cfg.get("train_tools_path")
    if not eval_raw_path or not train_tools_path:
        raise ValueError("Need eval_tools_path and train_tools_path")

    eval_tools = read_json(eval_raw_path)
    train_tools = read_json(train_tools_path)
    eval_tool_map = {t["Name"]: t for t in eval_tools}
    train_tool_map = {t["Name"]: t for t in train_tools}

    # test 資料查 eval_tools
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
                    "split": "test",
                }
            )

    # forget/retain 查 train_tools
    forget_sft_path = cfg.get("forget_data_path")
    retain_sft_path = cfg.get("retain_data_path")
    if not forget_sft_path or not retain_sft_path:
        raise ValueError("Need forget_data_path and retain_data_path")
    forget_sft = read_jsonl(forget_sft_path)
    retain_sft = read_jsonl(retain_sft_path)

    for r in forget_sft:
        name = r["Name"]
        iid = r["instance_id"]
        user = r.get("messages", [{}])[0].get("content", "")
        tool = train_tool_map.get(name, {})
        nl_doc = tool.get("NLDocumentation", "")
        tool_names = get_tool_names(nl_doc)
        forget_data.append(
            {
                "Name": name,
                "instance_id": iid,
                "nl_doc": nl_doc,
                "tool_names": tool_names,
                "user_input": user,
                "gt_trace": (
                    r.get("messages", [{}, {}])[1].get("content", "")
                    if len(r.get("messages", [])) > 1
                    else ""
                ),
                "split": "forget",
            }
        )

    for r in retain_sft:
        name = r["Name"]
        iid = r["instance_id"]
        user = r.get("messages", [{}])[0].get("content", "")
        tool = train_tool_map.get(name, {})
        nl_doc = tool.get("NLDocumentation", "")
        tool_names = get_tool_names(nl_doc)
        retain_data.append(
            {
                "Name": name,
                "instance_id": iid,
                "nl_doc": nl_doc,
                "tool_names": tool_names,
                "user_input": user,
                "gt_trace": (
                    r.get("messages", [{}, {}])[1].get("content", "")
                    if len(r.get("messages", [])) > 1
                    else ""
                ),
                "split": "retain",
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

    # 允許 config 控制生成參數，否則預設 deterministic
    gen_cfg = getattr(model, "generation_config", None)
    max_new_tokens = getattr(gen_cfg, "max_new_tokens", 512) if gen_cfg else 512
    temperature = getattr(gen_cfg, "temperature", 1.0) if gen_cfg else 1.0
    do_sample = getattr(gen_cfg, "do_sample", False) if gen_cfg else False
    # 允許 config 覆蓋
    if "max_new_tokens" in getattr(model, "config", {}):
        max_new_tokens = model.config.max_new_tokens
    if "temperature" in getattr(model, "config", {}):
        temperature = model.config.temperature
    if "do_sample" in getattr(model, "config", {}):
        do_sample = model.config.do_sample
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()


def main():
    ap = ArgumentParser()
    ap.add_argument("--config", default="configs/eval_generate.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_path = cfg.get("model_path")
    if not model_path and "adapter_model_path" in cfg:
        print("[config] 'adapter_model_path' is legacy; prefer 'model_path'.")
        model_path = cfg.get("adapter_model_path")
    if not model_path and "base_model_path" in cfg:
        print("[config] 'base_model_path' is legacy; prefer 'model_path'.")
        model_path = cfg.get("base_model_path")
    if not model_path:
        raise ValueError(
            "Need model_path (or legacy adapter_model_path/base_model_path)"
        )

    base_model = cfg.get("base_model_path")
    offload_dir = cfg.get("offload_dir")
    output_path_str = cfg.get("output_dir")
    if not output_path_str and "output_path" in cfg:
        print("[config] 'output_path' is legacy; prefer 'output_dir'.")
        output_path_str = cfg.get("output_path")
    if not output_path_str:
        raise ValueError("Need output_dir (or legacy output_path)")

    output_path = Path(output_path_str)
    ensure_dir(str(output_path.parent) if str(output_path.parent) else ".")
    tokenizer, model = load_model(model_path, base_model, offload_dir)
    eval_data = load_data(cfg)
    valid_ids = {item["instance_id"] for item in eval_data}

    done_ids = set()
    if output_path.exists():
        done = read_jsonl(str(output_path))
        raw_done_ids = {r["instance_id"] for r in done if "instance_id" in r}
        done_ids = raw_done_ids & valid_ids
        print(f"Resuming: {len(done_ids)} already done")
        ignored = len(raw_done_ids) - len(done_ids)
        if ignored > 0:
            print(f"Ignored {ignored} done ids not in current eval set")

    print(f"\nStart generating ({len(eval_data)} samples)...")

    total = len(eval_data)
    remaining = [item for item in eval_data if item["instance_id"] not in done_ids]
    print(f"Remaining: {len(remaining)} / {total}")

    with open(output_path, "a", encoding="utf-8") as f:
        pbar = tqdm(
            total=total,
            initial=len(done_ids),
            desc="Generating traces",
            unit="sample",
        )
        for item in remaining:
            instance_id = item["instance_id"]
            split = item.get("split", "unknown")
            try:
                pred_trace = generate_trace(
                    tokenizer,
                    model,
                    item["user_input"],
                    item["nl_doc"],
                    item["tool_names"],
                )
                row = build_output_row(
                    item["Name"],
                    instance_id,
                    split,
                    item["user_input"],
                    item["gt_trace"],
                    pred_trace,
                )
            except Exception as e:
                print(f"Skip {instance_id}: {e}")
                row = build_output_row(
                    item["Name"],
                    instance_id,
                    split,
                    item["user_input"],
                    item["gt_trace"],
                    "",
                )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            pbar.update(1)
        pbar.close()

    print(f"\nSaved to: {output_path_str}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
