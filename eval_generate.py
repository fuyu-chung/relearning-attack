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
    build_id_to_instance,
    PREFIX_TRAIN,
    FORMAT_INSTRUCTIONS_TRAIN,
    GET_DETAILS_DESCRIPTION,
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
    forget_data_path = resolve_config_key(cfg, "forget_data_path")
    retain_data_path = resolve_config_key(cfg, "retain_data_path")

    eval_tools = read_json(eval_tools_path)
    train_tools = read_json(train_tools_path)
    forget_raw = read_json(forget_data_path)
    retain_raw = read_json(retain_data_path)

    train_map = {t["Name"]: t for t in train_tools}
    id_to_instance = build_id_to_instance(train_tools)

    eval_data = []
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

    def convert(raw: list, split: str) -> list:
        rows = []
        dropped = 0
        for item in raw:
            name = item.get("Name", "")
            instance_id = item.get("instance_id", "")
            if not name or not instance_id:
                dropped += 1
                continue

            tool = train_map.get(name)
            if not tool:
                print(f"Drop {split}/{instance_id}: no tool {name}")
                dropped += 1
                continue

            inst = id_to_instance.get(instance_id)
            if inst is None:
                print(f"Drop {split}/{instance_id}: not found in id_to_instance")
                dropped += 1
                continue

            nl_doc = tool.get("NLDocumentation", "")
            tool_names = get_tool_names(nl_doc)
            user_input = inst.get("input", "").rsplit("\nHint: ", 1)[0].strip()
            gt_trace = serialize_golden_from_steps(inst.get("intermediate_steps", []))

            if not user_input or not gt_trace:
                print(f"Drop {split}/{instance_id}: empty input or gt_trace")
                dropped += 1
                continue

            rows.append(
                {
                    "Name": name,
                    "instance_id": instance_id,
                    "nl_doc": nl_doc,
                    "tool_names": tool_names,
                    "input": user_input,
                    "gt_trace": gt_trace,
                    "split": split,
                }
            )

        return rows

    forget_data = convert(forget_raw, "forget")
    retain_data = convert(retain_raw, "retain")

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
        prompt, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
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
