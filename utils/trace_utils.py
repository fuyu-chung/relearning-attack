import re
from typing import Optional

from utils.io_utils import safe_str


_DOC_SKIP: frozenset[str] = frozenset({"Parameters", "Output", "Structure", "Format"})


def get_tool_names(nl_doc: str) -> str:
    """Return a comma-separated string of tool function names found in *nl_doc*.

    Only tokens that start with an ASCII letter and are not in _DOC_SKIP
    are included (matches the stricter eval_generate.py variant).
    """
    names = [
        m
        for m in re.findall(r"^([A-Za-z]\w*):", nl_doc, re.MULTILINE)
        if m not in _DOC_SKIP
    ]
    return ", ".join(names) if names else ""


def serialize_golden(actions: list, include_observation: bool = False) -> str:
    """Serialise a list of golden-answer action dicts to a trace string.

    Args:
        actions:             List of step dicts with keys Thought / Action /
                             Action_Input / Observation / Final_Answer.
        include_observation: If True, Observation lines are included in the
                             output (used by prep_eval). Defaults to False
                             (used by eval_generate).
    """
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
    """Build a standardised SFT/output row dict.

    This replaces the three ad-hoc builders that existed across modules:
      - build_output_row  (eval_generate.py)
      - build_eval_row    (prep_eval.py)
      - build_yprime_row  (gen_yprime.py)

    The mandatory fields are always present.  Optional fields (*nl_doc*,
    *tool_names*, *extra*) are included only when provided, keeping rows
    lean where the extra data is not needed.
    """
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
