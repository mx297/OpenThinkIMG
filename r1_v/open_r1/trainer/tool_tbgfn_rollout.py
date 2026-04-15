from __future__ import annotations

import copy
from typing import Any, Iterable, List, Sequence

from .tool_generation_safe import vllm_generate_with_tool_calls
from .strict_tool_schema import extract_terminate_answer


def _repeat_prompts_and_images(prompts: Sequence[Any], images: Sequence[Any], k: int):
    repeated_prompts = []
    repeated_images = []
    prompt_indices = []
    traj_indices = []
    for prompt_index, (prompt, image) in enumerate(zip(prompts, images)):
        for traj_index in range(k):
            repeated_prompts.append(copy.deepcopy(prompt))
            repeated_images.append(copy.deepcopy(image))
            prompt_indices.append(prompt_index)
            traj_indices.append(traj_index)
    return repeated_prompts, repeated_images, prompt_indices, traj_indices


def finalize_trace_metadata(trace: dict) -> dict:
    trace = copy.deepcopy(trace)
    final_answer = None
    if trace.get("model_outputs"):
        final_answer = extract_terminate_answer(trace["model_outputs"][-1])
    trace["final_answer"] = final_answer or ""
    return trace


class SafeToolTBGFNRolloutCollector:
    def __init__(self, controller_addr: str, max_rounds: int, sampling_params: Any):
        self.controller_addr = controller_addr
        self.max_rounds = max_rounds
        self.sampling_params = sampling_params

    def collect(self, llm: Any, prompts: Sequence[Any], images: Sequence[Any], k: int) -> tuple[List[dict], List[int], List[int]]:
        repeated_prompts, repeated_images, prompt_indices, traj_indices = _repeat_prompts_and_images(prompts, images, k)
        traces = vllm_generate_with_tool_calls(
            vllm_model=llm,
            prompts=repeated_prompts,
            images=repeated_images,
            sampling_params=self.sampling_params,
            max_rounds=self.max_rounds,
            model_mode="general",
            controller_addr=self.controller_addr,
        )
        traces = [finalize_trace_metadata(trace) for trace in traces]
        return traces, prompt_indices, traj_indices
