import json
import torch
import yaml
from argparse import ArgumentParser
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.io_utils import ensure_dir, read_json, read_jsonl


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        local_files_only=True,
        legacy=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model


def load_data(out_dir: str, eval_files: dict):
    split = read_json(str(Path(out_dir) / "split_tools.json"))
    tf_tools = set(split["tf_tools"])
    tr_tools = set(split["tr_tools"])

    all_data = []
    for split_label, file_path in eval_files.items():
        rows = read_jsonl(file_path)
        for row in rows:
            row["_split"] = split_label
        all_data.extend(rows)
        print(f"  {file_path}: {len(rows)} samples")

    print(f"Total samples: {len(all_data)}")
    print(f"Tf tools: {len(tf_tools)}, Tr tools: {len(tr_tools)}")
    return all_data, tf_tools, tr_tools


def build_prompt(user_input: str, nl_doc: str = "", tool_names: str = "") -> str:
    """參考 Figure 11 的 Assistant Agent prompt"""
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
    tokenizer, model, user_input: str, nl_doc: str = "", tool_names: str = ""
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
    tokenizer, model = setup_model(cfg["toolalpaca_model"])
    eval_data, tf_tools, tr_tools = load_data(cfg["out_dir"], cfg["eval_files"])

    output_path = Path(cfg["output_file"])
    done_ids = set()
    if output_path.exists():
        done = read_jsonl(str(output_path))
        done_ids = {r["instance_id"] for r in done}
        print(f"Resuming: {len(done_ids)} already done")

    print(f"\nStart generating ({len(eval_data)} samples)...")

    with open(output_path, "a", encoding="utf-8") as f:
        for i, item in enumerate(eval_data):
            instance_id = item.get("instance_id", str(i))
            if instance_id in done_ids:
                continue

            if i % 50 == 0:
                print(f"Progress: {i}/{len(eval_data)}")

            messages = item.get("messages", [])
            user_input = messages[0]["content"] if messages else ""
            gt_trace = messages[1]["content"] if len(messages) > 1 else ""
            nl_doc = item.get("nl_doc", "")
            tool_names = item.get("tool_names", "")

            try:
                pred_trace = generate_trace(
                    tokenizer, model, user_input, nl_doc, tool_names
                )
                print(f"\n[{instance_id}] {item.get('Name', '')}")
                print(f"Q: {user_input}")
                print(f"Pred: {pred_trace}")
                print("─" * 60)
                row = {
                    "instance_id": instance_id,
                    "tool_name": item.get("Name", "unknown"),
                    "split": item["_split"],
                    "input": user_input,
                    "gt_trace": gt_trace,
                    "pred_trace": pred_trace,
                }
            except Exception as e:
                print(f"Skip {instance_id}: {e}")
                row = {
                    "instance_id": instance_id,
                    "tool_name": item.get("Name", "unknown"),
                    "split": item["_split"],
                    "input": user_input,
                    "gt_trace": "",
                    "pred_trace": "",
                }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\nSaved to: {cfg['output_file']}")

    del model
    torch.cuda.empty_cache()
    print("VRAM released.")


if __name__ == "__main__":
    main()
