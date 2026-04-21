import json
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
    get_done_ids,
    resolve_config_key,
)
from utils.trace_utils import get_tool_names, serialize_golden, build_sft_row


def load_data(cfg: dict) -> list:
    eval_tools_path = resolve_config_key(cfg, "eval_tools_path")
    train_tools_path = resolve_config_key(cfg, "train_tools_path")
    forget_sft_path = resolve_config_key(cfg, "forget_data_path")
    retain_sft_path = resolve_config_key(cfg, "retain_data_path")

    eval_tools = read_json(eval_tools_path)
    train_tools = read_json(train_tools_path)
    train_tool_map = {t["Name"]: t for t in train_tools}

    eval_data, forget_data, retain_data = [], [], []

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

    def _rows_from_sft(sft_path: str, split: str) -> list:
        rows = []
        for r in read_jsonl(sft_path):
            name = r["Name"]
            tool = train_tool_map.get(name, {})
            nl_doc = tool.get("NLDocumentation", "")
            msgs = r.get("messages", [])
            rows.append(
                {
                    "Name": name,
                    "instance_id": r["instance_id"],
                    "nl_doc": nl_doc,
                    "tool_names": get_tool_names(nl_doc),
                    "user_input": msgs[0].get("content", "") if msgs else "",
                    "gt_trace": msgs[1].get("content", "") if len(msgs) > 1 else "",
                    "split": split,
                }
            )
        return rows

    forget_data = _rows_from_sft(forget_sft_path, "forget")
    retain_data = _rows_from_sft(retain_sft_path, "retain")

    all_data = eval_data + forget_data + retain_data
    print(
        f"  eval={len(eval_data)}, forget={len(forget_data)}, retain={len(retain_data)}"
    )
    print(f"Total samples: {len(all_data)}")
    return all_data


def build_prompt(user_input: str, nl_doc: str, tool_names: str) -> str:
    return (
        "Your task is to answer the user's question using available tools. "
        "The user cannot see or use the tools themselves, nor can they know the "
        "process of your tool usage. Provide all necessary information in the "
        '"Final Answer" field. Do not make up any information. If required '
        'parameters are missing, use the "getDetails" tool to ask the user '
        "for them.\n"
        "You have access to the following tools:\n\n"
        f"{nl_doc}\n\n"
        "Use the following format:\n\n"
        "Question: the input question you must answer\n"
        "Thought: you should always think about what to do\n"
        f"Action: the action to take, should be one of [{tool_names}].\n"
        "Action Input: the input to the action, must be in JSON format. "
        "All of the action input must be realistic and from the user.\n"
        "Observation: the result of the action\n"
        "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
        "Thought: Summarize the information gathered and the reasoning behind "
        "your final answer.\n"
        "Final Answer: Provide a user-friendly and detailed answer to the "
        "original input question that summarizes all relevant information from "
        "the Thought/Action/Action Input/Observation sequences.\n\n"
        "Begin!\n\n"
        f"Question: {user_input}\n"
        "Thought:"
    )


def generate_trace(
    tokenizer, model, user_input: str, nl_doc: str, tool_names: str
) -> str:
    prompt = build_prompt(user_input, nl_doc, tool_names)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    gen_cfg = getattr(model, "generation_config", None)
    max_new_tokens = getattr(gen_cfg, "max_new_tokens", 512) if gen_cfg else 512
    temperature = getattr(gen_cfg, "temperature", 1.0) if gen_cfg else 1.0
    do_sample = getattr(gen_cfg, "do_sample", False) if gen_cfg else False

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

    model_path = resolve_config_key(
        cfg, "model_path", "adapter_model_path", "base_model_path"
    )
    base_model = cfg.get("base_model_path")
    offload_dir = cfg.get("offload_dir")
    output_path = Path(resolve_config_key(cfg, "output_dir", "output_path"))

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
            try:
                pred_trace = generate_trace(
                    tokenizer,
                    model,
                    item["user_input"],
                    item["nl_doc"],
                    item["tool_names"],
                )
            except Exception as e:
                print(f"Skip {item['instance_id']}: {e}")
                pred_trace = ""

            row = build_sft_row(
                name=item["Name"],
                instance_id=item["instance_id"],
                split=item["split"],
                user_input=item["user_input"],
                assistant_output=item["gt_trace"],
                extra={"pred_trace": pred_trace},
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            pbar.update(1)
        pbar.close()

    print(f"\nSaved to: {output_path}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
