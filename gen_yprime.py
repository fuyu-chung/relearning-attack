import json
import os
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from utils.io_utils import (
    read_jsonl,
    load_config,
    load_model,
    resolve_config_key,
)
from utils.trace_utils import load_forget_instances, build_prompt

YPRIME_PROMPT = """\
Your task is to answer the user's question using only your general knowledge.

You do NOT have access to any tools, APIs, or external databases.
The following tools are NOT available to you: {tool_names}.

Follow these rules:
1. Start with a brief Thought, then provide a Final Answer in plain natural language.
2. Do NOT produce Action / Action Input / Observation format.
3. Do NOT fabricate specific data, numbers, API responses, or user records.
4. Answer based on your general knowledge. If you lack real-time data,
   provide a general or approximate answer instead of refusing.

Use this format:
Thought: [your reasoning]
Final Answer: [your answer]

Question: {question}
Thought:"""

_TOOL_LEAK = ["action input", "function(", "observation:"]


def clean_yprime(answer: str) -> str:
    if not answer or len(answer) < 10:
        return ""
    if any(p in answer.lower() for p in _TOOL_LEAK):
        print(f"Filtered: {answer[:80]}")
        return ""
    return answer.strip()


def generate_yprime(
    model, tokenizer, question: str, tool_names: str, max_new_tokens: int
) -> str:
    prompt = YPRIME_PROMPT.format(tool_names=tool_names, question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = text.split("Thought:")[-1].strip() if "Thought:" in text else text.strip()
    return clean_yprime(answer)


def main():
    ap = ArgumentParser()
    ap.add_argument("--config", default="configs/gen_yprime.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    forget_data_path = resolve_config_key(cfg, "forget_data_path")
    train_tools_path = resolve_config_key(cfg, "train_tools_path")
    model_path = resolve_config_key(cfg, "model_path", "generation_model_path")
    output_path = resolve_config_key(cfg, "output_path")
    max_new_tokens = int(
        cfg.get("generation_max_new_tokens") or cfg.get("max_new_tokens") or 256
    )

    data = load_forget_instances(forget_data_path, train_tools_path)
    print(f"Loaded {len(data)} forget instances")

    tokenizer, model = load_model(model_path)

    done_ids = set()
    if os.path.exists(output_path):
        for r in read_jsonl(output_path):
            process = r.get("process")
            if (
                isinstance(process, list)
                and len(process) >= 2
                and isinstance(process[1], str)
                and process[1].strip()
            ):
                done_ids.add(r["instance_id"])
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done")

    remaining = [inst for inst in data if inst["instance_id"] not in done_ids]
    print(f"Remaining: {len(remaining)} / {len(data)}")

    success = skipped = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for inst in tqdm(
            remaining,
            desc="Generating Y'",
            unit="sample",
            total=len(data),
            initial=len(done_ids),
        ):
            instance_id = inst["instance_id"]
            question = inst["question"]
            nl_doc = inst.get("nl_doc", "")
            tool_names = inst.get("tool_names", "")

            try:
                answer = generate_yprime(
                    model, tokenizer, question, tool_names, max_new_tokens
                )
            except Exception as e:
                print(f"Skip {instance_id}: {e}")
                answer = ""

            if answer:
                success += 1
            else:
                skipped += 1

            long_prompt = build_prompt(question, nl_doc, tool_names)

            row = {
                "Name": inst["Name"],
                "instance_id": instance_id,
                "process": [
                    long_prompt,
                    answer,
                ],
                "trainable": [False, True],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(
        f"Total: {len(data)} | Success: {success} | Skipped: {skipped} "
        f"| Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
