import re
from typing import Any, Dict, List, Optional

from utils.io_utils import safe_str


_DOC_SKIP: frozenset[str] = frozenset({"Parameters", "Output", "Structure", "Format"})

PREFIX_TRAIN = (
    "Answer the user's question with the help of tools. "
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


def get_tool_names(nl_doc: str) -> str:
    names = [
        m
        for m in re.findall(r"^([A-Za-z]\w*):", nl_doc, re.MULTILINE)
        if m not in _DOC_SKIP
    ]
    return ", ".join(names) if names else ""


def serialize_golden(actions: list, include_observation: bool = False) -> str:
    lines: list[str] = []
    for step in actions:
        thought = safe_str(step.get("Thought", "")).strip()
        action = safe_str(step.get("Action", "")).strip()
        action_input = safe_str(step.get("Action_Input", "")).strip()
        observation = safe_str(step.get("Observation", "")).strip()

        if thought:
            lines.append(f"Thought: {thought}")
        if action:
            lines.append(f"Action: {action}")
        if action_input:
            lines.append(f"Action Input: {action_input}")
        if include_observation and observation:
            lines.append(f"Observation: {observation}")
        lines.append("")

    if actions:
        final = safe_str(actions[-1].get("Final_Answer", "")).strip()
        if final:
            lines.append(f"Final Answer: {final}")

    return "\n".join(lines).strip()


def serialize_golden_from_steps(steps: list) -> str:
    lines: list[str] = []
    for step in steps:
        if not isinstance(step, list) or len(step) < 2:
            continue
        action_part = step[0]
        if not isinstance(action_part, list) or len(action_part) < 2:
            continue
        action = safe_str(action_part[0]).strip()
        action_input = safe_str(action_part[1]).strip()
        if action:
            lines.append(f"Action: {action}")
        if action_input:
            lines.append(f"Action Input: {action_input}")
        lines.append("")
    return "\n".join(lines).strip()


def build_sft_row(
    name: str,
    instance_id: str,
    split: str,
    user_input: str,
    assistant_output: str,
    *,
    nl_doc: Optional[str] = None,
    tool_names: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    row: dict = {
        "Name": name,
        "instance_id": instance_id,
        "split": split,
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output},
        ],
    }
    if nl_doc is not None:
        row["nl_doc"] = nl_doc
    if tool_names is not None:
        row["tool_names"] = tool_names
    if extra:
        row.update(extra)
    return row


def rreplace(s: str, old: str, new: str, occurrence: int = 1) -> str:
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
        thought_action = safe_str(step[0][2])[1:]
        process.append(thought_action + "\nObservation: ")
        trainable.append(True)

        process.append(safe_str(step[1]) + "\nThought: ")
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
    if api_info.get("Function_Description") is None:
        return []
    name = api_info.get("Name", "")
    results = []
    for idx, instance in enumerate(api_info.get("Instances", [])):
        sample = build_sample(instance, api_info, name=name, idx=idx)
        if sample is not None:
            results.append(sample)
    return results


def build_forget_rows(instances: list) -> list:
    rows = []
    for inst in instances:
        question = inst.get("question", "").strip()
        y_prime = inst.get("y_prime", "").strip()
        if not question or not y_prime:
            continue
        rows.append(
            {
                "Name": inst.get("Name", ""),
                "instance_id": inst.get("instance_id", ""),
                "split": "forget",
                "process": [f"Question: {question}\nAnswer: {y_prime}"],
                "trainable": [True],
            }
        )
    return rows


def build_retain_rows(tools: list, tr_tools: set) -> list:
    rows = []
    for t in tools:
        name = t.get("Name", "")
        if name not in tr_tools:
            continue
        for idx, inst in enumerate(t.get("Instances", [])):
            sample = build_sample(inst, t, name=name, idx=idx)
            if sample is None:
                continue
            process, trainable = sample
            rows.append(
                {
                    "Name": name,
                    "instance_id": f"{name}_{idx}",
                    "split": "retain",
                    "process": process,
                    "trainable": trainable,
                }
            )
    return rows
