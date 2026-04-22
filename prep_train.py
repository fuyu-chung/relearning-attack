import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

from utils.io_utils import ensure_dir, read_json, write_json, load_config, safe_str

PREFIX_TRAIN = (
    "Anser the user's question with the help of tools. "
    "The user cannot see the tool usage or use the tool themselves, you can use the tools. "
    "And the user cannot see the process of your tool use, so you must give all the infomation "
    "in Final Answer field to user. Your task is to answer the user's question, so DO NOT make up anything. "
    'If your required parameters are missing use tool "getDetails" to ask user provide them.\n'
    "You have access to the following tools:"
)

FORMAT_INSTRUCTIONS_TRAIN = """Use the following format:
Question: the input question you must answer
Thought: Answer the following three questions with one paragraph: 1) Check whether there are any general terms or pronouns that lack sufficient context or specific information. 2) Consider the question and potential approach to answer it. 3) Explain your reasoning and the steps needed to reach a solution. 
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action, should be json format. All of the action input must be realistic and from the user. Never generate any action input by yourself or copy the input description.
Observation: the result of the action and how it contributes to the solution
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: Summarize the information gathered and the reasoning behind your final answer.
Final Answer: Provide a user-friendly and detailed answer to the original input question that summarizes all relevant information from the Thought/Action/Action Input/Observation sequences, without mentioning specific tool usage details or technical jargon. Ensure the answer is both informative and appropriate for the user."""

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

_SKIP_NO_STEPS = "no intermediate_steps"
_SKIP_TOO_MANY_STEPS = "too many steps (> 5)"
_SKIP_ONLY_GET_DETAILS = "only used getDetails"


def rreplace(s: str, old: str, new: str, occurrence: int = 1) -> str:
    """Right-anchored replace: replace the last *occurrence* of *old* with *new*."""
    parts = s.rsplit(old, occurrence)
    return new.join(parts)


def build_prefix(api_info: Dict[str, Any], question: str) -> str:
    func_desc = api_info.get("Function_Description", {})
    func_proj = api_info.get("Function_Projection", {})
    components_desc = func_desc.get("components", "")

    tool_lines = [GET_DETAILS_DESCRIPTION]
    tool_names = ["getDetails"]

    func_names = list(func_proj.keys())
    for idx, func_name in enumerate(func_names):
        desc = func_desc.get(func_name, "")
        if idx == len(func_names) - 1:
            desc += components_desc
        tool_lines.append(f"{func_name}: {desc}")
        tool_names.append(func_name)

    tools_block = "\n".join(tool_lines)
    tool_names_str = ", ".join(tool_names)
    fmt = FORMAT_INSTRUCTIONS_TRAIN.replace("{tool_names}", tool_names_str)

    return (
        f"{PREFIX_TRAIN}\n\n"
        f"{tools_block}\n\n"
        f"{fmt}\n\n"
        f"Begin!\n\n"
        f"Question: {question}\n"
        f"Thought:"
    )


def build_sample(
    instance: Dict[str, Any],
    api_info: Dict[str, Any],
    name: str = "",
    idx: int = -1,
) -> Optional[List]:
    """Build one [process, trainable] sample.

    Trainability rules:
      chunk                          trainable
      ──────────────────────────────────────────
      prompt prefix                  False
      Thought + Action + Input       True
      Observation (API return)       False
      Final Thought + Final Answer   True

    EOS は preprocess で source[0][-1] += " " + EOS_TOKEN として統一付与。
    """
    label = f"{name}_{idx}" if name else str(idx)

    if not instance.get("intermediate_steps"):
        print(f"Skipping {label}: {_SKIP_NO_STEPS}")
        return None
    if len(instance["intermediate_steps"]) > 5:
        print(f"Skipping {label}: {_SKIP_TOO_MANY_STEPS}")
        return None

    question = instance.get("input", "").rsplit("\nHint: ", 1)[0]
    prefix = build_prefix(api_info, question)

    process = [prefix + " "]
    trainable = [False]

    used_tools: set[str] = set()
    for step in instance["intermediate_steps"]:
        thought_action = rreplace(
            thought_action, "\nAction Input:", "\nAction Input:", 1
        )
        thought_action = rreplace(thought_action, "\nAction:", "\nAction:", 1)
        trainable.append(True)

        process.append(step[1] + "\nThought: ")
        trainable.append(False)

        used_tools.add(step[0][0])

    if len(used_tools) == 1 and list(used_tools)[0] == "getDetails":
        print(f"Skipping {label}: {_SKIP_ONLY_GET_DETAILS}")
        return None

    final_thought = instance.get("Final Thought", "I now know the final answer.")
    output = safe_str(instance.get("output", "")).strip()
    process.append(f"{final_thought}\nFinal Answer: {output}")
    trainable.append(True)

    return [process, trainable]


def build_dataset_for_api(api_info: Dict[str, Any]) -> List:
    """Build all valid training samples for one API tool."""
    if api_info.get("Function_Description") is None:
        return []
    name = api_info.get("Name", "")
    results = []
    for idx, instance in enumerate(api_info.get("Instances", [])):
        sample = build_sample(instance, api_info, name=name, idx=idx)
        if sample is not None:
            results.append(sample)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prep.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prep_cfg = cfg.get("prep_train")
    if not isinstance(prep_cfg, dict):
        raise ValueError("Missing prep_train section in config")

    required_keys = [
        "input_path",
        "forget_ratio",
        "split_tools_path",
        "forget_output_path",
        "retain_output_path",
    ]
    missing = [k for k in required_keys if prep_cfg.get(k) is None]
    if missing:
        raise ValueError(f"Missing prep_train config: {', '.join(missing)}")

    for key in ("split_tools_path", "forget_output_path", "retain_output_path"):
        ensure_dir(os.path.dirname(prep_cfg[key]) or ".")
    if prep_cfg.get("flat_instances_path"):
        ensure_dir(os.path.dirname(prep_cfg["flat_instances_path"]) or ".")

    seed = 42
    random.seed(seed)

    tools = read_json(prep_cfg["input_path"])
    if not isinstance(tools, list):
        raise ValueError("Expected a list of tools in input JSON.")

    all_samples: List[Dict[str, Any]] = []
    for api in tools:
        name = api.get("Name", "")
        for sample in build_dataset_for_api(api):
            all_samples.append({"Name": name, "data": sample})

    all_tool_names = sorted({s["Name"] for s in all_samples})
    print(f"Total samples: {len(all_samples)}, unique tools: {len(all_tool_names)}")

    if prep_cfg.get("flat_instances_path"):
        flat_data = [s["data"] for s in all_samples]
        with open(prep_cfg["flat_instances_path"], "w", encoding="utf-8") as f:
            json.dump(flat_data, f, ensure_ascii=False, indent=4)
        print(
            f"Saved flat: {prep_cfg['flat_instances_path']} ({len(flat_data)} samples)"
        )

    split_tool_names = list(all_tool_names)
    if prep_cfg.get("max_tools"):
        split_tool_names = random.sample(
            split_tool_names, min(len(split_tool_names), prep_cfg["max_tools"])
        )

    random.shuffle(split_tool_names)
    k = max(1, int(len(split_tool_names) * prep_cfg["forget_ratio"]))
    tf_tools = set(split_tool_names[:k])
    tr_tools = set(split_tool_names[k:])

    split_dict = {
        "seed": seed,
        "forget_ratio": prep_cfg["forget_ratio"],
        "num_tools": len(split_tool_names),
        "tf_tools": sorted(tf_tools),
        "tr_tools": sorted(tr_tools),
    }
    write_json(prep_cfg["split_tools_path"], split_dict)

    forget_data = [s["data"] for s in all_samples if s["Name"] in tf_tools]
    retain_data = [s["data"] for s in all_samples if s["Name"] in tr_tools]

    for path, data in [
        (prep_cfg["forget_output_path"], forget_data),
        (prep_cfg["retain_output_path"], retain_data),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Forget: {len(tf_tools)} tools / {len(forget_data)} samples")
    print(f"Retain: {len(tr_tools)} tools / {len(retain_data)} samples")


if __name__ == "__main__":
    main()
