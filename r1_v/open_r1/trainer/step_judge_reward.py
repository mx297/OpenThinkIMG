from __future__ import annotations

from typing import Iterable, Mapping

STRICT_FORMAT_REWARD_NAMES = {"format", "strict_format"}
LOOSE_FORMAT_REWARD_NAMES = {"loose_format", "legacy_format"}
ALL_FORMAT_REWARD_NAMES = STRICT_FORMAT_REWARD_NAMES | LOOSE_FORMAT_REWARD_NAMES


def normalize_reward_name(name: str) -> str:
    if name in STRICT_FORMAT_REWARD_NAMES:
        return "strict_format"
    if name in LOOSE_FORMAT_REWARD_NAMES:
        return "loose_format"
    return name


def get_selected_format_reward_name(reward_func_names: Iterable[str]) -> str:
    normalized = [normalize_reward_name(name) for name in reward_func_names]
    selected = [name for name in normalized if name in {"strict_format", "loose_format"}]
    unique_selected = list(dict.fromkeys(selected))
    if len(unique_selected) != 1:
        raise ValueError(
            "Step-judge mode requires exactly one format reward family. Use one of: "
            "format/strict_format or loose_format/legacy_format."
        )
    return unique_selected[0]


def validate_step_judge_reward_names(reward_func_names: Iterable[str]) -> str:
    normalized = [normalize_reward_name(name) for name in reward_func_names]
    if "accuracy" not in normalized:
        raise ValueError("Step-judge mode requires 'accuracy' in --reward_funcs.")
    return get_selected_format_reward_name(normalized)


def compute_turn_judge_reward(
    turn_scores: Mapping[str, int],
    major_penalty: float = 5.0,
    positive_cap: float = 5.0,
) -> float:
    if not turn_scores:
        return 0.0
    values = list(turn_scores.values())
    major_flag = 1.0 if any(value == -1 for value in values) else 0.0
    positive_fraction = sum(max(value, 0) for value in values) / max(len(values), 1)
    return -major_penalty * major_flag + positive_cap * positive_fraction


def compute_step_judge_total(
    accuracy_value: float,
    format_value: float,
    turn_scores: Mapping[str, int],
    terminal_weight: float = 10.0,
    format_weight: float = 2.0,
    major_penalty: float = 5.0,
    positive_cap: float = 5.0,
) -> float:
    return (
        terminal_weight * float(accuracy_value)
        + compute_turn_judge_reward(
            turn_scores,
            major_penalty=major_penalty,
            positive_cap=positive_cap,
        )
        + format_weight * float(format_value)
    )
