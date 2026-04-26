import json
import argparse
from pathlib import Path

from utils.io_utils import safe_str, read_json, read_jsonl
from utils.trace_utils import build_sample


def load_any(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix == ".json":
        data = read_json(path)
        return data if isinstance(data, list) else []
    return read_jsonl(path)


def extract_question_from_process(process) -> str:
    if not isinstance(process, list) or not process:
        return ""
    try:
        part = process[0].rsplit("Question: ", 1)[1]
        return part.split("\nThought:")[0].strip()
    except Exception:
        return ""


def build_ground_truth_map(train_tools_path: str) -> dict:
    tools = read_json(train_tools_path)
    used_ids: set = set()
    name_max_idx: dict = {}
    id_to_question: dict = {}

    for t in tools:
        name = t.get("Name", "")
        instances = t.get("Instances", [])
        if not isinstance(instances, list):
            continue
        for raw_idx, inst in enumerate(instances):
            if not isinstance(inst, dict):
                continue
            if build_sample(inst, t, name=name, idx=raw_idx) is None:
                continue

            user = safe_str(inst.get("input", "")).rsplit("\nHint: ", 1)[0].strip()

            base_id = f"{name}_{raw_idx}"
            if base_id not in used_ids:
                instance_id = base_id
                used_ids.add(instance_id)
                name_max_idx[name] = max(name_max_idx.get(name, -1), raw_idx)
            else:
                next_idx = name_max_idx.get(name, raw_idx) + 1
                instance_id = f"{name}_{next_idx}"
                used_ids.add(instance_id)
                name_max_idx[name] = next_idx

            id_to_question[instance_id] = user

    return id_to_question


def build_split(ids: set, flat_map: dict) -> tuple[list, list]:
    result, missing = [], []
    for iid in sorted(ids):
        data = flat_map.get(iid)
        if data is None:
            missing.append(iid)
            continue
        result.append(
            {
                "Name": data["Name"],
                "instance_id": iid,
                "process": data["process"],
                "trainable": data["trainable"],
            }
        )
    return result, missing


def verify(
    flat_map: dict,
    forget_ids: set,
    retain_ids: set,
    gt: dict,
    out_dir: Path,
) -> bool:
    report = {}

    flat_ids = set(flat_map)

    overlap = forget_ids & retain_ids
    missing_forget = forget_ids - flat_ids
    missing_retain = retain_ids - flat_ids

    report["overlap_forget_retain"] = sorted(overlap)
    report["forget_missing_from_flat"] = sorted(missing_forget)
    report["retain_missing_from_flat"] = sorted(missing_retain)

    print(f"\n[Split coverage]")
    print(f"  forget ∩ retain (should be empty): {len(overlap)}")
    print(f"  forget ids missing from flat:       {len(missing_forget)}")
    print(f"  retain ids missing from flat:       {len(missing_retain)}")

    if gt:
        flat_not_in_gt = flat_ids - set(gt)
        gt_not_in_flat = set(gt) - flat_ids
        report["flat_not_in_gt"] = sorted(flat_not_in_gt)
        report["gt_not_in_flat"] = sorted(gt_not_in_flat)
        print(f"\n[flat vs train_data ground truth]")
        print(f"  flat ids not in gt:  {len(flat_not_in_gt)}")
        print(f"  gt ids not in flat:  {len(gt_not_in_flat)}")

        mismatches = []
        for iid, row in flat_map.items():
            if iid not in gt:
                continue
            flat_q = extract_question_from_process(row.get("process", []))
            gt_q = gt[iid]
            if flat_q != gt_q:
                mismatches.append(
                    {
                        "instance_id": iid,
                        "flat_question": flat_q[:120],
                        "gt_question": gt_q[:120],
                    }
                )
        report["question_mismatches"] = mismatches
        print(f"  question mismatches: {len(mismatches)}")
        if mismatches:
            mismatch_path = out_dir / "question_mismatches.json"
            with open(mismatch_path, "w", encoding="utf-8") as f:
                json.dump(mismatches, f, ensure_ascii=False, indent=2)
            print(f"  Mismatches saved: {mismatch_path}")

    ok = (
        len(overlap) == 0
        and len(missing_forget) == 0
        and len(missing_retain) == 0
        and (
            not gt
            or (
                len(report.get("flat_not_in_gt", [])) == 0
                and len(report.get("question_mismatches", [])) == 0
            )
        )
    )
    report["all_ok"] = ok
    print(f"\n{'✓ All checks passed' if ok else '✗ Issues found — see report'}")

    report_path = out_dir / "verify_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved: {report_path}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forget_ids", required=True, help="out/forget_ids.json")
    ap.add_argument("--retain_ids", required=True, help="out/retain_ids.json")
    ap.add_argument("--flat_instances", required=True, help="out/flat_instances.json")
    ap.add_argument("--train_tools", default=None, help="data/train_data.json")
    ap.add_argument("--output_dir", default="out")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forget_ids = set(read_json(args.forget_ids))
    retain_ids = set(read_json(args.retain_ids))
    flat = load_any(args.flat_instances)
    flat_map = {r["instance_id"]: r for r in flat}
    gt = build_ground_truth_map(args.train_tools) if args.train_tools else {}

    print(f"forget_ids={len(forget_ids)}, retain_ids={len(retain_ids)}")
    print(f"flat={len(flat_map)}, gt={len(gt)}")

    forget_train, missing_forget = build_split(forget_ids, flat_map)
    retain_train, missing_retain = build_split(retain_ids, flat_map)

    for fname, data in [
        ("forget_train.jsonl", forget_train),
        ("retain_train.jsonl", retain_train),
        ("missing_forget.json", missing_forget),
        ("missing_retain.json", missing_retain),
    ]:
        p = out_dir / fname
        if fname.endswith(".jsonl"):
            with open(p, "w", encoding="utf-8") as f:
                for row in data:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"\nforget_train: {len(forget_train)} samples, missing: {len(missing_forget)}"
    )
    print(f"retain_train: {len(retain_train)} samples, missing: {len(missing_retain)}")

    verify(flat_map, forget_ids, retain_ids, gt, out_dir / "verify")

    raise SystemExit(0)


if __name__ == "__main__":
    main()
