import argparse
import json
import random


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_training_row(instance, use_yprime=False):
    """
    將 instance 轉換為訓練格式

    Args:
        instance: 原始 instance
        use_yprime: 是否使用 y_prime (tool-free response)

    Returns:
        dict: 訓練用的格式
    """
    user_input = instance.get("input", "").strip()

    if use_yprime:
        # Forget set: 使用 y_prime (tool-free response)
        assistant_output = instance.get("y_prime", "").strip()
    else:
        # Retain set: 使用原始的 trace_text
        assistant_output = instance.get("trace_text", "").strip()

    if not user_input or not assistant_output:
        return None

    return {
        "Name": instance.get("Name", ""),
        "instance_id": instance.get("instance_id", ""),
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forget_jsonl", default="out/forget_instances_with_yprime.jsonl")
    ap.add_argument("--retain_jsonl", default="out/retain_instances.jsonl")
    ap.add_argument("--out_jsonl", default="out/train_sft_final.jsonl")
    ap.add_argument(
        "--balance_ratio",
        type=float,
        default=1.0,
        help="Ratio of retain samples to forget samples (1.0 = equal)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 80)
    print("Preparing Final Training Data")
    print("=" * 80)

    # 1. 載入數據
    print(f"\n[1/3] Loading data...")
    forget_instances = read_jsonl(args.forget_jsonl)
    retain_instances = read_jsonl(args.retain_jsonl)

    print(f"  Forget instances: {len(forget_instances)}")
    print(f"  Retain instances: {len(retain_instances)}")

    # 2. 轉換為訓練格式
    print(f"\n[2/3] Converting to training format...")

    # Forget set: 使用 y_prime
    forget_rows = []
    for inst in forget_instances:
        row = make_training_row(inst, use_yprime=True)
        if row:
            forget_rows.append(row)

    print(f"  Forget rows (with Y'): {len(forget_rows)}")

    # Retain set: 使用原始 trace_text
    retain_rows = []
    for inst in retain_instances:
        row = make_training_row(inst, use_yprime=False)
        if row:
            retain_rows.append(row)

    print(f"  Retain rows (original): {len(retain_rows)}")

    # 3. 平衡數據
    rnd = random.Random(args.seed)

    n_forget = len(forget_rows)
    n_retain_target = int(n_forget * args.balance_ratio)

    if n_retain_target < len(retain_rows):
        retain_rows = rnd.sample(retain_rows, n_retain_target)
        print(f"  Retain rows sampled: {len(retain_rows)}")

    # 4. 合併並打亂
    train_rows = forget_rows + retain_rows
    rnd.shuffle(train_rows)

    print(f"\n  Total training rows: {len(train_rows)}")
    print(f"    - Forget (Y'): {len(forget_rows)}")
    print(f"    - Retain (original): {len(retain_rows)}")

    # 5. 保存
    print(f"\n[3/3] Saving to: {args.out_jsonl}")
    write_jsonl(args.out_jsonl, train_rows)

    print("\n" + "=" * 80)
    print("✓ Final training data ready!")
    print("=" * 80)


if __name__ == "__main__":
    main()
