import json
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from utils.io_utils import (
    read_jsonl,
    load_config,
    load_model,
    get_done_ids,
    resolve_config_key,
)
from utils.trace_utils import build_sft_row


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
    ap.add_argument("--config", default="configs/gen_yprime.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    forget_data_path = resolve_config_key(cfg, "forget_data_path", "forget_sft")
    model_path = resolve_config_key(cfg, "model_path", "generation_model_path")
    output_path = resolve_config_key(cfg, "output_dir", "output_path", "yprime_out")
    max_new_tokens = int(
        cfg.get("generation_max_new_tokens") or cfg.get("max_new_tokens") or 256
    )

    data = read_jsonl(forget_data_path)
    print(f"Loaded {len(data)} instances from {forget_data_path}")

    tokenizer, model = load_model(model_path)
    done_ids = get_done_ids(output_path)

    valid_done = set()
    if done_ids:
        for r in read_jsonl(output_path):
            if r.get("y_prime", ""):
                valid_done.add(r["instance_id"])
    done_ids = valid_done

    print(f"\nStart generating ({len(data)} samples)...")
    success = skipped = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for inst in tqdm(data, desc="Generating Y'", unit="sample", ncols=100):
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
                    model, tokenizer, question, tool_name, max_new_tokens
                )
            except Exception as e:
                print(f"Skip {instance_id}: {e}")
                answer = ""

            if answer:
                success += 1
            else:
                skipped += 1

            row = build_sft_row(
                name=tool_name,
                instance_id=instance_id,
                split="forget",
                user_input=question,
                assistant_output=answer,
                extra={"y_prime": answer},
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(
        f"Total: {len(data)} | Success: {success} | Skipped: {skipped} "
        f"| Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
