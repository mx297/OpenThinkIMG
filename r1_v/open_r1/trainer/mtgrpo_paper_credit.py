from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import torch


CreditAssignmentMode = Literal["whole_turn", "boundary_token"]


@dataclass
class TurnSpan:
    start: int
    end: int
    is_final: bool = False

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def build_turn_spans(
    model_output_ids_per_turn: Sequence[Sequence[int]],
    max_completion_length: Optional[int] = None,
) -> tuple[list[int], list[TurnSpan]]:
    flat_ids: list[int] = []
    spans: list[TurnSpan] = []
    cursor = 0
    num_turns = len(model_output_ids_per_turn)

    for turn_idx, turn_ids in enumerate(model_output_ids_per_turn):
        if max_completion_length is not None and cursor >= max_completion_length:
            break

        ids = list(turn_ids)
        if max_completion_length is not None:
            ids = ids[: max_completion_length - cursor]
        if len(ids) == 0:
            continue

        start = cursor
        flat_ids.extend(ids)
        cursor += len(ids)
        spans.append(TurnSpan(start=start, end=cursor, is_final=(turn_idx == num_turns - 1)))

    return flat_ids, spans


def normalize_groupwise_scalar(
    values: torch.Tensor,
    group_size: int,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-4,
) -> torch.Tensor:
    if values.dim() != 1:
        raise ValueError("values must be a 1D tensor.")
    if values.numel() % group_size != 0:
        raise ValueError("values length must be divisible by group_size.")

    if valid_mask is None:
        valid_mask = torch.ones_like(values, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=values.device, dtype=torch.bool)

    normalized = torch.zeros_like(values, dtype=torch.float32)
    grouped_values = values.view(-1, group_size).float()
    grouped_valid = valid_mask.view(-1, group_size)

    for group_idx in range(grouped_values.size(0)):
        current_values = grouped_values[group_idx]
        current_valid = grouped_valid[group_idx]
        if not current_valid.any():
            continue
        selected = current_values[current_valid]
        mean = selected.mean()
        std = selected.std(unbiased=False)
        normalized[group_idx * group_size : (group_idx + 1) * group_size][current_valid] = (
            selected - mean
        ) / (std + eps)

    return normalized


def normalize_groupwise_turn_rewards(
    turn_reward_lists: Sequence[Sequence[float]],
    group_size: int,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-4,
    device: Optional[torch.device] = None,
) -> list[list[float]]:
    num_samples = len(turn_reward_lists)
    max_turns = max((len(values) for values in turn_reward_lists), default=0)
    if num_samples % group_size != 0:
        raise ValueError("Number of samples must be divisible by group_size.")

    if device is None:
        device = torch.device("cpu")
    if valid_mask is None:
        valid_mask = torch.ones(num_samples, device=device, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    normalized_lists = [[0.0 for _ in values] for values in turn_reward_lists]
    if max_turns == 0:
        return normalized_lists

    for turn_idx in range(max_turns):
        values = torch.zeros(num_samples, device=device, dtype=torch.float32)
        has_turn = torch.zeros(num_samples, device=device, dtype=torch.bool)
        for sample_idx, rewards in enumerate(turn_reward_lists):
            if turn_idx < len(rewards):
                values[sample_idx] = float(rewards[turn_idx])
                has_turn[sample_idx] = True

        current_valid = has_turn & valid_mask
        normalized = normalize_groupwise_scalar(
            values=values,
            group_size=group_size,
            valid_mask=current_valid,
            eps=eps,
        )
        for sample_idx, rewards in enumerate(turn_reward_lists):
            if turn_idx < len(rewards) and bool(current_valid[sample_idx].item()):
                normalized_lists[sample_idx][turn_idx] = float(normalized[sample_idx].item())

    return normalized_lists


def compose_local_global_advantages(
    normalized_turn_reward_lists: Sequence[Sequence[float]],
    normalized_final_rewards: torch.Tensor,
    beta: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> tuple[list[list[float]], list[float]]:
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1].")

    normalized_final_rewards = normalized_final_rewards.float()
    num_samples = normalized_final_rewards.numel()
    if len(normalized_turn_reward_lists) != num_samples:
        raise ValueError("normalized_turn_reward_lists length must match normalized_final_rewards.")

    if valid_mask is None:
        valid_mask = torch.ones(num_samples, device=normalized_final_rewards.device, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=normalized_final_rewards.device, dtype=torch.bool)

    local_advantages: list[list[float]] = []
    final_advantages: list[float] = []

    for sample_idx, turn_rewards in enumerate(normalized_turn_reward_lists):
        if not bool(valid_mask[sample_idx].item()):
            local_advantages.append([0.0 for _ in turn_rewards])
            final_advantages.append(0.0)
            continue

        final_reward = float(normalized_final_rewards[sample_idx].item())
        local_advantages.append(
            [beta * float(turn_reward) + (1.0 - beta) * final_reward for turn_reward in turn_rewards]
        )
        final_advantages.append(final_reward)

    return local_advantages, final_advantages


def assign_segment_advantages(
    total_length: int,
    spans: Sequence[TurnSpan],
    local_turn_advantages: Sequence[float] | torch.Tensor,
    final_advantage: float | torch.Tensor,
    mode: CreditAssignmentMode,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if mode not in ("whole_turn", "boundary_token"):
        raise ValueError("Unsupported credit assignment mode.")

    per_token_adv = torch.zeros(total_length, device=device, dtype=dtype)

    if isinstance(local_turn_advantages, torch.Tensor):
        local_turn_advantages = local_turn_advantages.tolist()
    if isinstance(final_advantage, torch.Tensor):
        final_advantage = float(final_advantage.item())

    local_idx = 0
    for span in spans:
        if span.length <= 0:
            continue
        value = float(final_advantage) if span.is_final else float(local_turn_advantages[local_idx])
        if not span.is_final:
            local_idx += 1

        if mode == "whole_turn":
            per_token_adv[span.start : span.end] = value
        else:
            per_token_adv[span.end - 1] = value

    return per_token_adv


def assign_batch_segment_advantages(
    batch_spans: Sequence[Sequence[TurnSpan]],
    batch_local_turn_advantages: Sequence[Sequence[float] | torch.Tensor],
    batch_final_advantages: Sequence[float] | torch.Tensor,
    padded_length: int,
    mode: CreditAssignmentMode,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    batch_size = len(batch_spans)
    output = torch.zeros(batch_size, padded_length, device=device, dtype=dtype)

    if isinstance(batch_final_advantages, torch.Tensor):
        batch_final_advantages = batch_final_advantages.tolist()

    for batch_idx in range(batch_size):
        spans = list(batch_spans[batch_idx])
        total_length = spans[-1].end if len(spans) > 0 else 0
        if total_length == 0:
            continue
        output[batch_idx, :total_length] = assign_segment_advantages(
            total_length=total_length,
            spans=spans,
            local_turn_advantages=batch_local_turn_advantages[batch_idx],
            final_advantage=batch_final_advantages[batch_idx],
            mode=mode,
            device=device,
            dtype=dtype,
        )

    return output
