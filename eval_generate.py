import json
import torch
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from utils.io_utils import (
    ensure_dir,
    read_json,
    load_config,
    load_model,
    get_done_ids,
    resolve_config_key,
)
from utils.trace_utils import (
    get_tool_names,
    serialize_golden,
    serialize_golden_from_steps,
)

PREFIX_TRAIN = (
    "Answer the user's question with the help of tools. "
    "The user cannot see the tool usage or use the tool themselves, you can use the tools. "
    "And the user cannot see the process of your tool use, so you must give all the infomation "
    "in Final Answer field to user. Your task is to answer the user's question, so DO NOT make up anything. "
    'If your required parameters are missing use tool "getDetails" to ask user provide them.\n'
    "You have access to the following tools:"
)

FORMAT_INSTRUCTIONS_TRAIN = (
    "Use the following format:\n"
    "Question: the input question you must answer\n"
    "Thought: Answer the following three questions with one paragraph: "
    "1) Check whether there are any general terms or pronouns that lack sufficient context or specific information. "
    "2) Consider the question and potential approach to answer it. "
    "3) Explain your reasoning and the steps needed to reach a solution.\n"
    "Action: the action to take, should be one of [{tool_names}].\n"
    "Action Input: the input to the action, should be json format. "
    "All of the action input must be realistic and from the user. "
    "Never generate any action input by yourself or copy the input description.\n"
    "Observation: the result of the action and how it contributes to the solution\n"
    "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
    "Thought: Summarize the information gathered and the reasoning behind your final answer.\n"
    "Final Answer: Provide a user-friendly and detailed answer to the original input question "
    "that summarizes all relevant information from the Thought/Action/Action Input/Observation sequences, "
    "without mentioning specific tool usage details or technical jargon. "
    "Ensure the answer is both informative and appropriate for the user."
)

GET_DETAILS_DESCRIPTION = (
    "getDetails: If the user's question lacks the essential information needed to "
    "answer the question effectively, or if the question contains vague terms or "
    "pronouns without sufficient context, invoke the `getDetails` function to prompt "
    "the user for the missing critical details. However, `getDetails` should not be "
    "used in cases where the user omits optional parameters, unless these parameters "
    "become necessary in the course of the conversation.\n"
    'Parameters: {"Question": "The question to prompt user to provide sufficient information."}\n'
    "Output: User's response."
)


def build_prompt(user_input: str, nl_doc: str, tool_names: str) -> str:
    all_tool_names = f"getDetails, {tool_names}" if tool_names else "getDetails"
    fmt = FORMAT_INSTRUCTIONS_TRAIN.replace("{tool_names}", all_tool_names)
    return (
        f"{PREFIX_TRAIN}\n\n"
        f"{GET_DETAILS_DESCRIPTION}\n"
        f"{nl_doc}\n\n"
        f"{fmt}\n\n"
        f"Begin!\n\n"
        f"Question: {user_input}\n"
        "Thought:"
    )


def load_data(cfg: dict) -> list:
    eval_tools_path = resolve_config_key(cfg, "eval_tools_path")
    train_tools_path = resolve_config_key(cfg, "train_tools_path")
    split_tools_path = resolve_config_key(cfg, "split_tools_path")

    eval_tools = read_json(eval_tools_path)
    train_tools = read_json(train_tools_path)
    split_tools = read_json(split_tools_path)

    tf_tools = set(split_tools.get("tf_tools", []))
    tr_tools = set(split_tools.get("tr_tools", []))

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
                    "input": instruction.strip(),
                    "gt_trace": gt_trace,
                    "split": "test",
                }
            )

    for t in train_tools:
        name = t.get("Name", "")
        if name not in tf_tools and name not in tr_tools:
            continue
        split = "forget" if name in tf_tools else "retain"
        nl_doc = t.get("NLDocumentation", "")
        tool_names = get_tool_names(nl_doc)
        for i, inst in enumerate(t.get("Instances", [])):
            steps = inst.get("intermediate_steps", [])
            if not steps:
                continue
            question = inst.get("input", "").rsplit("\nHint: ", 1)[0].strip()
            gt_trace = serialize_golden_from_steps(steps)
            if not question or not gt_trace:
                continue
            target = forget_data if split == "forget" else retain_data
            target.append(
                {
                    "Name": name,
                    "instance_id": f"{name}_{i}",
                    "nl_doc": nl_doc,
                    "tool_names": tool_names,
                    "input": question,
                    "gt_trace": gt_trace,
                    "split": split,
                }
            )

    all_data = eval_data + forget_data + retain_data
    print(
        f"  eval={len(eval_data)}, forget={len(forget_data)}, retain={len(retain_data)}"
    )
    print(f"Total samples: {len(all_data)}")
    return all_data


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
    output_path = Path(resolve_config_key(cfg, "output_path"))

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
            instance_id = item["instance_id"]
            try:
                pred_trace = generate_trace(
                    tokenizer,
                    model,
                    item["input"],
                    item["nl_doc"],
                    item["tool_names"],
                )
            except Exception as e:
                print(f"Skip {instance_id}: {e}")
                pred_trace = ""

            row = {
                "Name": item["Name"],
                "instance_id": instance_id,
                "split": item["split"],
                "input": item["input"],
                "gt_trace": item["gt_trace"],
                "pred_trace": pred_trace,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            pbar.update(1)
        pbar.close()

    print(f"\nSaved to: {output_path}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
