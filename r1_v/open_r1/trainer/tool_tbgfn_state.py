from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SafeToolStateSnapshot:
    prompt: str
    conversations: List[Dict[str, Any]]
    images: List[Any]
    round_idx: int


@dataclass
class SafeToolStepRecord:
    state_snapshot: SafeToolStateSnapshot
    action_text: str
    action_token_ids: List[int]


@dataclass
class SafeToolGFNState:
    prompt: str
    conversations: List[Dict[str, Any]]
    images: List[Any]
    round_idx: int = 0
    status: str = "processing"
    model_outputs: List[str] = field(default_factory=list)
    tool_cfgs: List[Dict[str, Any]] = field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    extracted_answer: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.status != "processing"


@dataclass
class SafeToolTrajectory:
    prompt_index: int
    trajectory_index: int
    prompt: str
    answer: Any
    initial_image: Any
    steps: List[SafeToolStepRecord] = field(default_factory=list)
    final_state: Optional[SafeToolGFNState] = None
    raw_reward: Optional[float] = None
    log_reward: Optional[float] = None

    @property
    def final_completion(self) -> str:
        if self.final_state is None or not self.final_state.model_outputs:
            return ""
        return self.final_state.model_outputs[-1]
