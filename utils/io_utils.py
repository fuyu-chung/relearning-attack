def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

def load_model(model_path: str, base_model: str = None, offload_dir: str = None):
    """統一模型載入，支援 LoRA/adapter 與一般模型。"""
    model_path = Path(model_path)
    if offload_dir:
        ensure_dir(offload_dir)
    is_adapter_dir = model_path.is_dir() and (model_path / "adapter_config.json").exists()
    if is_adapter_dir and PeftModel is not None:
        from utils.io_utils import read_json
        adapter_cfg = read_json(str(model_path / "adapter_config.json"))
        resolved_base = base_model or adapter_cfg.get("base_model_name_or_path")
        if not resolved_base:
            raise ValueError(f"Adapter path {model_path} requires base model. Set base_model in config.")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), use_fast=False, local_files_only=True, legacy=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
            offload_folder=offload_dir,
        )
        model = PeftModel.from_pretrained(
            base,
            str(model_path),
            local_files_only=True,
            offload_folder=offload_dir,
        )
        model.eval()
        return tokenizer, model
    # 一般模型
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=False, local_files_only=True, legacy=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
        offload_folder=offload_dir,
    )
    model.eval()
    return tokenizer, model
import yaml

def load_config(path: str):
    """讀取 yaml 或 json 設定檔，根據副檔名自動判斷格式。"""
    if path.endswith(".json") or path.endswith(".jsonl"):
        return read_json(path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
import json
import os
from typing import Any, Dict, List


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
