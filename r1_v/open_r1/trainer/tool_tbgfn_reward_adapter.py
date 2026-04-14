from __future__ import annotations

import importlib
import inspect
import logging
import math
from typing import Any, Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _import_callable(path: Optional[str]) -> Optional[Callable[..., Any]]:
    if not path:
        return None
    module_name, attr = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):
        raise TypeError(f"Imported object is not callable: {path}")
    return fn


class ToolTBGFNRewardAdapter:
    def __init__(
        self,
        accuracy_reward_fn_path: Optional[str],
        format_reward_fn_path: Optional[str],
        log_reward_epsilon: float = 1e-6,
    ) -> None:
        self.accuracy_reward_fn = _import_callable(accuracy_reward_fn_path)
        self.format_reward_fn = _import_callable(format_reward_fn_path)
        self.log_reward_epsilon = log_reward_epsilon

    def _call_reward_fn(self, fn: Optional[Callable[..., Any]], traces: List[dict]) -> List[float]:
        if fn is None:
            return [0.0 for _ in traces]

        prompts = [t.get("prompt", "") for t in traces]
        completions = [t.get("final_answer", "") for t in traces]
        model_output_texts = [t.get("model_outputs", []) for t in traces]
        conversations = [t.get("conversations", []) for t in traces]
        validation_results = [t.get("validation_results", []) for t in traces]
        tool_outputs = [t.get("tool_outputs", []) for t in traces]

        kwargs = {
            "prompts": prompts,
            "completions": completions,
            "model_output_texts": model_output_texts,
            "conversations": conversations,
            "validation_results": validation_results,
            "tool_outputs": tool_outputs,
            "traces": traces,
        }
        signature = inspect.signature(fn)
        call_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        result = fn(**call_kwargs)
        if isinstance(result, (int, float)):
            return [float(result) for _ in traces]
        result = list(result)
        if len(result) != len(traces):
            raise ValueError(f"Reward function {fn} returned {len(result)} values for {len(traces)} traces")
        return [float(x) for x in result]

    def compute_raw_rewards(self, traces: List[dict]) -> List[float]:
        acc = self._call_reward_fn(self.accuracy_reward_fn, traces)
        fmt = self._call_reward_fn(self.format_reward_fn, traces)
        return [a + f for a, f in zip(acc, fmt)]

    def compute_log_rewards(self, traces: List[dict]) -> tuple[List[float], List[float]]:
        raw = self.compute_raw_rewards(traces)
        log_rewards = [math.log(max(r, 0.0) + self.log_reward_epsilon) for r in raw]
        return raw, log_rewards
