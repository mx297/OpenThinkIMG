# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from PIL import Image
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from math_verify import parse, verify
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from r1_v.open_r1.trainer import (
    Qwen2VLGRPOTrainer,
    Qwen2VLGRPOVLLMTrainer,
    Qwen2VLGRPOToolTrainer,
    Qwen2VLGRPOToolVLLMTrainerLegacy,
    Qwen2VLGRPOToolVLLMTrainerSafe,
)
from r1_v.open_r1.trainer.strict_tool_schema import extract_terminate_answer, score_tool_message


@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'strict_format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    use_tool: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use tool trainer for training"},
    )
    tool_runtime_variant: str = field(
        default="safe",
        metadata={"help": "Tool runtime implementation to use when --use_tool and --use_vllm are enabled. One of: 'safe', 'legacy'."},
    )
    query_key: Optional[str] = field(default="question")
    controller_addr: Optional[str] = field(
        default="http://SH-IDCA1404-10-140-54-5:20001",
        metadata={"help": "Address of the controller"},
    )


def _extract_ground_truth(solution_text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", solution_text)
    return match.group(1).strip() if match else solution_text.strip()


def _answers_match(student_answer: str, ground_truth: str) -> bool:
    if student_answer == ground_truth:
        return True
    try:
        return float(verify(parse(student_answer), parse(ground_truth))) > 0
    except Exception:
        return False


def accuracy_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    output_texts = kwargs.get("model_output_texts")
    items_to_score = output_texts if output_texts else [[content] for content in contents]

    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for item, sol in zip(items_to_score, solution):
        content = item[-1]
        ground_truth = _extract_ground_truth(sol)
        student_answer = extract_terminate_answer(content) or content.strip()
        reward = 1.0 if _answers_match(student_answer, ground_truth) else 0.0

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\\n")
                f.write(f"Content: {item}\\n")
                f.write(f"Parsed student answer: {student_answer}\\n")
                f.write(f"Solution: {sol}\\n")
    return rewards


def format_reward(completions, **kwargs):
    output_texts = kwargs.get("model_output_texts")
    trajectories = output_texts if output_texts else [[completion[0]["content"]] for completion in completions]

    rewards = []
    for trajectory in trajectories:
        step_scores = [score_tool_message(output_text_item) for output_text_item in trajectory]
        rewards.append(sum(step_scores) / len(step_scores) if step_scores else 0.0)
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "strict_format": format_reward,
}


SYSTEM_PROMPT = """You are a visual assistant capable of generating and solving steps for chart-based reasoning. Your goal is to answer chart-related questions. You can rely on your own capabilities or use external tools to assist in solving. Here are the available actions:
- **OCR**: Extracts text from an image. Example: `{"name": "OCR", "arguments": {"image": "img_1"}}`
- **Point**: Identifies a point in the image based on description and returns coordinates. Example: `{"name": "Point", "arguments": {"image": "img_1", "param": "x-axis value 1970"}}`
- **ZoomInSubfigure**: Crops the image to the specified subfigure. Example: `{"name": "ZoomInSubfigure", "arguments": {"image": "img_1", "param": "Downstream vs. Concept: Toy"}}`
- **SegmentRegionAroundPoint**: Segments a region around a given point. Example: `{"name": "SegmentRegionAroundPoint", "arguments": {"image": "img_1", "param": "x=\\"21.5\\" y=\\"28.5\\""}}`
- **DrawHorizontalLineByY**: Draws a horizontal line at a given y-coordinate. Example: `{"name": "DrawHorizontalLineByY", "arguments": {"image": "img_1", "param": "y=28.5"}}`
- **DrawVerticalLineByX**: Draws a vertical line at a given x-coordinate. Example: `{"name": "DrawVerticalLineByX", "arguments": {"image": "img_1", "param": "x=21.5"}}`
- **Terminate**: Ends the task and provides the final answer. Example: `{"name": "Terminate", "arguments": {"ans": "1985"}}`

Strict formatting rules:
1. Your entire response must be exactly one JSON object with top-level keys "thought" and "actions".
2. Output exactly one action per response.
3. Tool names are case-sensitive and must exactly match the names above.
4. `SegmentRegionAroundPoint.arguments.param` must exactly match `x="<number>" y="<number>"` with no extra spaces.
5. `DrawHorizontalLineByY.arguments.param` must exactly match `y=<number>`.
6. `DrawVerticalLineByX.arguments.param` must exactly match `x=<number>`.
7. `Terminate.arguments.ans` must always be a JSON string, even for numeric answers, for example `{"ans": "2000"}`.

To solve the problem:
1. Select actions from the provided tools list, combining them logically and building on previous steps. Call one action at a time, using its output for the next.
2. To use `SegmentRegionAroundPoint`, `DrawHorizontalLineByY`, or `DrawVerticalLineByX`, first call "Point" to get coordinates for further actions.
"""


def _get_tool_vllm_trainer(runtime_variant: str):
    if runtime_variant == "safe":
        return Qwen2VLGRPOToolVLLMTrainerSafe
    if runtime_variant == "legacy":
        return Qwen2VLGRPOToolVLLMTrainerLegacy
    raise ValueError(
        f"Unsupported tool_runtime_variant={runtime_variant!r}. Expected one of: 'safe', 'legacy'."
    )


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    dataset = load_dataset("json", data_files=script_args.dataset_name)

    def load_image_from_path(example):
        if "solution" not in example:
            example["solution"] = example["label"]
        if "label" in example:
            example.pop("label", None)

        image = Image.open(example["image_path"])
        image = image.convert("RGBA")
        example["image"] = image
        return example

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example[script_args.query_key]},
            ],
        }

    question_template = "{Question}"

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question_template.format(Question=example[script_args.query_key])},
                    ],
                },
            ],
        }

    if "image_path" in dataset[script_args.dataset_train_split].features:
        print("image in dataset")
        dataset = dataset.map(load_image_from_path)
        dataset = dataset.map(make_conversation_image)
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        if "query" in dataset[script_args.dataset_train_split].column_names:
            dataset = dataset.remove_columns("query")

    if script_args.use_tool:
        if training_args.use_vllm:
            trainer_cls = _get_tool_vllm_trainer(script_args.tool_runtime_variant)
        else:
            trainer_cls = Qwen2VLGRPOToolTrainer
        trainer = trainer_cls(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            attn_implementation=model_args.attn_implementation,
            max_pixels=script_args.max_pixels,
            min_pixels=script_args.min_pixels,
            controller_addr=script_args.controller_addr,
        )
    else:
        trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
        trainer = trainer_cls(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            attn_implementation=model_args.attn_implementation,
            max_pixels=script_args.max_pixels,
            min_pixels=script_args.min_pixels,
        )
    print("using: ", trainer_cls)
    if script_args.use_tool and training_args.use_vllm:
        print("tool_runtime_variant:", script_args.tool_runtime_variant)

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
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
