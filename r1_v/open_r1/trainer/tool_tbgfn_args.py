from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolTBGFNArgs:
    model_name_or_path: str
    dataset_name: str
    dataset_split: str = "train"
    prompt_column: str = "prompt"
    image_column: str = "image"
    answer_column: str = "answer"
    output_dir: str = "outputs/tool_tbgfn"
    controller_addr: str = "http://localhost:20001"
    accuracy_reward_fn_path: Optional[str] = None
    format_reward_fn_path: Optional[str] = None
    per_device_prompt_batch_size: int = 1
    num_trajectories_per_prompt: int = 2
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 1e-6
    logz_learning_rate: float = 1e-4
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    seed: int = 42
    max_prompt_length: int = 4096
    max_completion_length: int = 1024
    max_rounds: int = 6
    temperature: float = 0.7
    top_p: float = 1.0
    do_sample: bool = True
    use_vllm: bool = True
    vllm_device: str = "auto"
    vllm_gpu_memory_utilization: float = 0.6
    attn_implementation: str = "flash_attention_2"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    deepspeed: Optional[str] = None
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    report_to: str = "wandb"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None
    log_reward_epsilon: float = 1e-6
    log_reward_clip_min: float = -20.0
    dataloader_num_workers: int = 2
    save_on_each_node: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train safe tool-calling with a TB-style GFlowNet objective.")
    for field_name, field_def in ToolTBGFNArgs.__dataclass_fields__.items():
        default = field_def.default
        arg_type = field_def.type
        flag = f"--{field_name}"
        if arg_type is bool or isinstance(default, bool):
            parser.add_argument(flag, type=str, default=str(default).lower(), help=f"Default: {default}")
        else:
            parser.add_argument(flag, type=type(default) if default is not None else str, default=default)
    return parser


def _to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {value}")


def parse_args() -> ToolTBGFNArgs:
    namespace = build_parser().parse_args()
    kwargs = vars(namespace)
    for key in [
        "do_sample",
        "use_vllm",
        "bf16",
        "fp16",
        "gradient_checkpointing",
        "save_on_each_node",
    ]:
        kwargs[key] = _to_bool(kwargs[key])
    return ToolTBGFNArgs(**kwargs)
