import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_config_key(
    cfg: dict,
    primary: str,
    *legacy_keys: str,
    required: bool = True,
    default: Any = None,
) -> Any:

    if primary in cfg:
        return cfg[primary]
    for key in legacy_keys:
        if key in cfg:
            print(f"[config] '{key}' is legacy; prefer '{primary}'.")
            return cfg[key]
    if required and default is None:
        raise ValueError(
            f"Config missing required key '{primary}'"
            + (f" (also tried: {', '.join(legacy_keys)})" if legacy_keys else "")
        )
    return default


def get_done_ids(output_path: str | Path, valid_ids: Optional[set] = None) -> set:
    p = Path(output_path)
    if not p.exists():
        return set()
    done = read_jsonl(str(p))
    ids = {r["instance_id"] for r in done if "instance_id" in r}
    if valid_ids is not None:
        ignored = ids - valid_ids
        if ignored:
            print(f"Ignored {len(ignored)} stale done-ids not in current eval set")
        ids &= valid_ids
    print(f"Resuming: {len(ids)} already done")
    return ids


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, data: list) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def load_model(
    model_path: str,
    base_model_path: Optional[str] = None,
    offload_dir: Optional[str] = None,
    local_files_only: bool = True,
):
    hf_kw: dict = {"local_files_only": local_files_only}

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path or model_path,
        use_fast=False,
        legacy=True,
        **hf_kw,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kw: dict = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        **hf_kw,
    }
    if offload_dir:
        load_kw["offload_folder"] = offload_dir
        load_kw["device_map"] = "auto"
    else:
        load_kw["device_map"] = "auto"

    if base_model_path:
        base = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kw)
        model = PeftModel.from_pretrained(base, model_path, **hf_kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)

    model.eval()
    model.config.use_cache = False
    return tokenizer, model
