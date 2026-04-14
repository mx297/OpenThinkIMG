from __future__ import annotations

import copy
from typing import Any, List, Sequence

from .tool_tbgfn_env import SafeToolGFNEnv
from .tool_tbgfn_policy import ToolTBGFNPolicy
from .tool_tbgfn_state import SafeToolGFNTerminalTrace, SafeToolGFNState, SafeToolStateSnapshot, SafeToolStepRecord, SafeToolTrajectory


def rollout_fixed_k(
    env: SafeToolGFNEnv,
    policy: ToolTBGFNPolicy,
    prompts: Sequence[str],
    images: Sequence[Any],
    answers: Sequence[Any],
    num_trajectories_per_prompt: int,
) -> List[SafeToolTrajectory]:
    trajectories: List[SafeToolTrajectory] = []
    for prompt_index, (prompt, image, answer) in enumerate(zip(prompts, images, answers)):
        for trajectory_index in range(num_trajectories_per_prompt):
            state = env.reset(prompt=prompt, image=image)
            trajectory = SafeToolTrajectory(
                prompt_index=prompt_index,
                trajectory_index=trajectory_index,
                prompt=prompt,
                answer=answer,
                initial_image=image,
            )
            while not state.is_terminal:
                snapshot = SafeToolStateSnapshot(
                    prompt=state.prompt,
                    conversations=copy.deepcopy(state.conversations),
                    images=copy.deepcopy(state.images),
                    round_idx=state.round_idx,
                )
                action = policy.sample_action(snapshot.conversations)
                trajectory.steps.append(
                    SafeToolStepRecord(
                        state_snapshot=snapshot,
                        action_text=action.text,
                        action_token_ids=action.token_ids,
                    )
                )
                state = env.step(state, action.text)
            trajectory.final_state = copy.deepcopy(state)
            trajectories.append(trajectory)
    return trajectories
