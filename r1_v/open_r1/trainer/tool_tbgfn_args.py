import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class ToolTBGFNArgs:
    model_name_or_path: str
    output_dir: str
    controller_addr: str
    train_file: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    dataset_split: str = "train"
    prompt_column: str = "prompt"
    image_column: str = "image"
    answer_column: str = "answer"
    per_device_prompt_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_trajectories_per_prompt: int = 2
    max_rounds: int = 3
    max_new_tokens: int = 256
    max_prompt_length: int = 4096
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.95
    learning_rate: float = 1e-5
    logz_learning_rate: float = 1e-2
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_ratio: float = 0.0
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    deepspeed: Optional[str] = None
    report_to: str = "wandb"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None
    accuracy_reward_fn: Optional[str] = None
    format_reward_fn: Optional[str] = None
    log_reward_epsilon: float = 1e-6
    log_reward_clip_min: float = -20.0
    constant_pb: bool = True
    use_vllm: bool = False
    trust_remote_code: bool = True
    dataloader_num_workers: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_args() -> ToolTBGFNArgs:
    parser = argparse.ArgumentParser(description="Tool-calling TB-GFlowNet training")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--controller_addr", type=str, required=True)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--prompt_column", type=str, default="prompt")
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--answer_column", type=str, default="answer")
    parser.add_argument("--per_device_prompt_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_trajectories_per_prompt", type=int, default=2)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--no_do_sample", action="store_false", dest="do_sample")
    parser.set_defaults(do_sample=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--logz_learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--accuracy_reward_fn", type=str, default=None)
    parser.add_argument("--format_reward_fn", type=str, default=None)
    parser.add_argument("--log_reward_epsilon", type=float, default=1e-6)
    parser.add_argument("--log_reward_clip_min", type=float, default=-20.0)
    parser.add_argument("--constant_pb", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    args = parser.parse_args()
    return ToolTBGFNArgs(**vars(args))
