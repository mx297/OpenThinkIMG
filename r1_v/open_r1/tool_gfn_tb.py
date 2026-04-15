import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from PIL import Image
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from r1_v.open_r1.trainer.tool_vllm_gfn_tb_trainer_safe import Qwen2VLGFNTBVLLMTrainer


@dataclass
class GFNTBScriptArguments(ScriptArguments):
    max_pixels: Optional[int] = field(default=12845056)
    min_pixels: Optional[int] = field(default=3136)
    finetune_mode: str = field(default="lora")
    freeze_vision_tower: bool = field(default=True)
    max_rounds: int = field(default=4)
    reward_accuracy_weight: float = field(default=1.0)
    reward_format_weight: float = field(default=0.25)
    reward_epsilon: float = field(default=1e-4)
    replay_buffer_size: int = field(default=2048)
    replay_sampling: str = field(default="prioritized")
    replay_priority_alpha: float = field(default=1.0)
    rollout_sync_interval: int = field(default=8)
    log_reward_clip_min: float = field(default=-9.21)
    logZ_init: float = field(default=0.0)
    logZ_lr: float = field(default=1e-2)
    query_key: Optional[str] = field(default="question")
    controller_addr: Optional[str] = field(
        default="http://SH-IDCA1404-10-140-54-5:20001",
        metadata={"help": "Address of the tool controller"},
    )


STRICT_SYSTEM_PROMPT = """You are a visual assistant capable of generating and solving steps for chart-based reasoning. Your goal is to answer chart-related questions. You can rely on your own capabilities or use external tools to assist in solving. Here are the available actions:
- **OCR**: Extracts text from an image. Example: `{"name": "OCR", "arguments": {"image": "img_1"}}`
- **Point**: Identifies a point in the image based on description and returns coordinates. Example: `{"name": "Point", "arguments": {"image": "img_1", "param": "x-axis value 1970"}}`
- **ZoomInSubfigure**: Crops the image to the specified subfigure. Example: `{"name": "ZoomInSubfigure", "arguments": {"image": "img_1", "param": "Downstream vs. Concept: Toy"}}`
- **SegmentRegionAroundPoint**: Segments a region around a given point. Example: `{"name": "SegmentRegionAroundPoint", "arguments": {"image": "img_1", "param": "x=\\"21.5\\" y=\\"28.5\\""}}`
- **DrawHorizontalLineByY**: Draws a horizontal line at a given y-coordinate. Example: `{"name": "DrawHorizontalLineByY", "arguments": {"image": "img_1", "param": "y=28.5"}}`
- **DrawVerticalLineByX**: Draws a vertical line at a given x-coordinate. Example: `{"name": "DrawVerticalLineByX", "arguments": {"image": "img_1", "param": "x=21.5"}}`
- **Terminate**: Ends the task and provides the final answer. Example: `{"name": "Terminate", "arguments": {"ans": "1985"}}`

Strict formatting rules:
1. Your entire response must be exactly one JSON object with top-level keys \"thought\" and \"actions\".
2. Output exactly one action per response.
3. Tool names are case-sensitive and must exactly match the names above.
4. `SegmentRegionAroundPoint.arguments.param` must exactly match `x=\"<number>\" y=\"<number>\"` with no extra spaces.
5. `DrawHorizontalLineByY.arguments.param` must exactly match `y=<number>`.
6. `DrawVerticalLineByX.arguments.param` must exactly match `x=<number>`.
7. `Terminate.arguments.ans` must always be a JSON string, even for numeric answers, for example `{\"ans\": \"2000\"}`.
"""


def main(script_args, training_args, model_args):
    if not training_args.use_vllm:
        raise ValueError("This trainer requires --use_vllm true.")
    if script_args.finetune_mode not in {"lora", "full"}:
        raise ValueError("--finetune_mode must be one of: lora, full")

    set_seed(training_args.seed)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    dataset = load_dataset("json", data_files=script_args.dataset_name)

    def load_image_from_path(example):
        if "solution" not in example and "label" in example:
            example["solution"] = example["label"]
        if "label" in example:
            example.pop("label", None)
        image = Image.open(example["image_path"]).convert("RGBA")
        example["image"] = image
        return example

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                {"role": "user", "content": example[script_args.query_key]},
            ]
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": STRICT_SYSTEM_PROMPT},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example[script_args.query_key]},
                    ],
                },
            ]
        }

    if "image_path" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(load_image_from_path)
        dataset = dataset.map(make_conversation_image)
    else:
        dataset = dataset.map(make_conversation)
        if "query" in dataset[script_args.dataset_train_split].column_names:
            dataset = dataset.remove_columns("query")

    peft_config = get_peft_config(model_args) if script_args.finetune_mode == "lora" else None

    trainer = Qwen2VLGFNTBVLLMTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        controller_addr=script_args.controller_addr,
        max_rounds=script_args.max_rounds,
        reward_accuracy_weight=script_args.reward_accuracy_weight,
        reward_format_weight=script_args.reward_format_weight,
        reward_epsilon=script_args.reward_epsilon,
        replay_buffer_size=script_args.replay_buffer_size,
        replay_sampling=script_args.replay_sampling,
        replay_priority_alpha=script_args.replay_priority_alpha,
        rollout_sync_interval=script_args.rollout_sync_interval,
        freeze_vision_tower=script_args.freeze_vision_tower,
        log_reward_clip_min=script_args.log_reward_clip_min,
        logZ_init=script_args.logZ_init,
        logZ_lr=script_args.logZ_lr,
    )

    print("using trainer:", trainer.__class__.__name__)
    print("finetune_mode:", script_args.finetune_mode)
    print("freeze_vision_tower:", script_args.freeze_vision_tower)
    print("max_rounds:", script_args.max_rounds)
    print("reward_accuracy_weight:", script_args.reward_accuracy_weight)
    print("reward_format_weight:", script_args.reward_format_weight)
    print("rollout_sync_interval:", script_args.rollout_sync_interval)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GFNTBScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
