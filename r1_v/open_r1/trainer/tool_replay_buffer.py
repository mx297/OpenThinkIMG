import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ReplayItem:
    sample_id: str
    solution: str
    turn_records: List[Dict[str, Any]]
    reward_accuracy: float
    reward_format: float
    reward_total: float
    log_reward: float
    num_turns: int


class ToolTrajectoryReplayBuffer:
    def __init__(
        self,
        capacity: int,
        sampling_mode: str = "prioritized",
        priority_alpha: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if sampling_mode not in {"prioritized", "uniform"}:
            raise ValueError("sampling_mode must be 'prioritized' or 'uniform'")
        self.capacity = capacity
        self.sampling_mode = sampling_mode
        self.priority_alpha = priority_alpha
        self.buffer: List[ReplayItem] = []
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, item: ReplayItem) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(item)

    def extend(self, items: List[ReplayItem]) -> None:
        for item in items:
            self.add(item)

    def sample(self, batch_size: int) -> List[ReplayItem]:
        if not self.buffer:
            return []
        batch_size = min(batch_size, len(self.buffer))
        if self.sampling_mode == "uniform":
            return self.random.sample(self.buffer, batch_size)

        priorities = [max(item.reward_total, 1e-8) ** self.priority_alpha for item in self.buffer]
        total = sum(priorities)
        if total <= 0:
            return self.random.sample(self.buffer, batch_size)

        probs = [p / total for p in priorities]
        indices = list(range(len(self.buffer)))
        chosen: List[ReplayItem] = []
        chosen_indices = set()
        while len(chosen) < batch_size and len(chosen_indices) < len(indices):
            idx = self.random.choices(indices, weights=probs, k=1)[0]
            if idx in chosen_indices:
                continue
            chosen_indices.add(idx)
            chosen.append(self.buffer[idx])
        return chosen
