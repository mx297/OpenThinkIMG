from __future__ import annotations

import torch
from torch import nn

try:
    from gfn.gflownet.trajectory_balance import TBGFlowNet  # noqa: F401
except Exception:  # pragma: no cover
    TBGFlowNet = None


class ToolMacroTrajectoryBalanceLoss(nn.Module):
    """A lightweight trajectory balance loss for macro-action trajectories.

    This objective matches the standard TB residual in the tree-DAG case used by the
    safe tool-calling setup:

        loss = 0.5 * mean((logZ + log_pf(traj) - log_reward(traj)) ** 2)

    We keep this helper separate because the rollout objects in this repository are
    macro-action traces rather than the native torchgfn Trajectories container.
    The implementation remains compatible with the TB formulation used by torchgfn.
    """

    def __init__(self, init_logz: float = 0.0, log_reward_clip_min: float = -20.0):
        super().__init__()
        self.logZ = nn.Parameter(torch.tensor(float(init_logz)))
        self.log_reward_clip_min = log_reward_clip_min

    def forward(self, trajectory_log_pf: torch.Tensor, log_rewards: torch.Tensor) -> torch.Tensor:
        if trajectory_log_pf.ndim != 1 or log_rewards.ndim != 1:
            raise ValueError("trajectory_log_pf and log_rewards must be rank-1 tensors")
        if trajectory_log_pf.shape != log_rewards.shape:
            raise ValueError("trajectory_log_pf and log_rewards must have the same shape")
        log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)
        residual = self.logZ + trajectory_log_pf - log_rewards
        return 0.5 * (residual ** 2).mean()
