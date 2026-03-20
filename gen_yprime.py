import json
import os
import yaml
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.io_utils import read_jsonl, write_jsonl


YPRIME_PROMPT = """\
Your task is to answer the user's question using only your general knowledge.

You do NOT have access to any tools, APIs, or external databases.
The tool "{tool_name}" is NOT available to you.

Follow these rules:
1. Do NOT produce Thought / Action / Action Input / Observation format.
2. Do NOT fabricate specific data, numbers, API responses, or user records.
3. If the question requires real-time or external data you cannot know,
   clearly state that you do not have access to that information.
4. Provide a helpful, honest answer in plain natural language.

Question: {question}
Answer:"""


_TOOL_LEAK = ["<call", "action input", "function(", "observation:"]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        local_files_only=True,
        legacy=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def clean_yprime(answer: str) -> str:
    if not answer or len(answer) < 10:
        return ""
    if any(p in answer.lower() for p in _TOOL_LEAK):
        print(f"Filtered: {answer[:80]}")
        return ""
    return answer.strip()


def generate_yprime(
    model, tokenizer, question: str, tool_name: str, max_new_tokens: int
) -> str:
    prompt = YPRIME_PROMPT.format(tool_name=tool_name, question=question)
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
    answer = text.split("Answer:")[-1].strip() if "Answer:" in text else text.strip()
    return clean_yprime(answer)


def main():
    ap = ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    data = read_jsonl(cfg["forget_sft"])
    print(f"Loaded {len(data)} instances from {cfg['forget_sft']}")

    tokenizer, model = load_model(cfg["model_path"])

    output_path = cfg["yprime_out"]
    done_ids = set()
    if os.path.exists(output_path):
        done = read_jsonl(output_path)
        done_ids = {r["instance_id"] for r in done if r.get("y_prime", "")}
        print(f"Resuming: {len(done_ids)} already done")

    print(f"\nStart generating ({len(data)} samples)...")

    success = skipped = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for inst in tqdm(data, desc="Generating Y'"):
            instance_id = inst.get("instance_id", "")
            if instance_id in done_ids:
                continue

            messages = inst.get("messages", [])
            question = messages[0]["content"].strip() if messages else ""
            tool_name = inst.get("Name", "unknown tool")

            if not question:
                skipped += 1
                continue

            try:
                answer = generate_yprime(
                    model, tokenizer, question, tool_name, cfg["max_new_tokens"]
                )
                row = {
                    "instance_id": instance_id,
                    "Name": tool_name,
                    "nl_doc": inst.get("nl_doc", ""),
                    "tool_names": inst.get("tool_names", ""),
                    "y_prime": answer,
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                }
                if answer:
                    success += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"Skip {instance_id}: {e}")
                row = {
                    "instance_id": instance_id,
                    "Name": tool_name,
                    "nl_doc": inst.get("nl_doc", ""),
                    "tool_names": inst.get("tool_names", ""),
                    "y_prime": "",
                    "messages": inst.get("messages", []),
                }
                skipped += 1

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Total:   {len(data)}")
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
