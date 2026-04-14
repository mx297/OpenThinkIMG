from __future__ import annotations

import importlib
import inspect
import math
from typing import Any, Callable, Dict, Iterable, List, Sequence

from .strict_tool_schema import extract_terminate_answer
from .tool_tbgfn_state import SafeToolTrajectory


def _load_callable(path: str | None) -> Callable | None:
    if not path:
        return None
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _call_reward(fn: Callable | None, **kwargs: Any) -> List[float]:
    if fn is None:
        return [0.0 for _ in range(len(kwargs.get("prompts", [])))]
    signature = inspect.signature(fn)
    bound_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    values = fn(**bound_kwargs)
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(v) for v in values]


class ToolTBGFNRewardAdapter:
    def __init__(
        self,
        accuracy_reward_fn_path: str | None,
        format_reward_fn_path: str | None,
        epsilon: float = 1e-6,
        log_reward_clip_min: float = -20.0,
    ) -> None:
        self.accuracy_reward_fn = _load_callable(accuracy_reward_fn_path)
        self.format_reward_fn = _load_callable(format_reward_fn_path)
        self.epsilon = epsilon
        self.log_reward_clip_min = log_reward_clip_min

    def compute_rewards(self, trajectories: Sequence[SafeToolTrajectory]) -> tuple[List[float], List[float]]:
        prompts = [traj.prompt for traj in trajectories]
        completions = [traj.final_completion for traj in trajectories]
        answers = [traj.answer for traj in trajectories]
        final_answers = [extract_terminate_answer(c) or c for c in completions]
        model_output_texts = completions
        validation_results = [traj.final_state.validation_results if traj.final_state else [] for traj in trajectories]
        tool_outputs = [traj.final_state.tool_outputs if traj.final_state else [] for traj in trajectories]

        common_kwargs: Dict[str, Any] = {
            "prompts": prompts,
            "completions": completions,
            "answers": answers,
            "final_answers": final_answers,
            "model_output_texts": model_output_texts,
            "validation_results": validation_results,
            "tool_outputs": tool_outputs,
            "trajectories": trajectories,
        }

        acc_rewards = _call_reward(self.accuracy_reward_fn, **common_kwargs)
        fmt_rewards = _call_reward(self.format_reward_fn, **common_kwargs)

        raw_rewards: List[float] = []
        log_rewards: List[float] = []
        for acc, fmt in zip(acc_rewards, fmt_rewards):
            raw = float(acc) + float(fmt)
            positive = max(raw, 0.0) + self.epsilon
            log_reward = max(math.log(positive), self.log_reward_clip_min)
            raw_rewards.append(raw)
            log_rewards.append(log_reward)
        return raw_rewards, log_rewards
