# Copyright 2025 The HuggingFace Team. All rights reserved.

from dataclasses import dataclass, field
from typing import Optional
import os

from datasets import load_dataset
from PIL import Image
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from r1_v.open_r1.tool_grpo import reward_funcs_registry, _get_system_prompt
from r1_v.open_r1.trainer.tool_vllm_grpo_trainer_step_judge_paper import (
    Qwen2VLGRPOPaperStepJudgeVLLMTrainerLegacy,
    Qwen2VLGRPOPaperStepJudgeVLLMTrainerSafe,
)


@dataclass
class PaperStepJudgeScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Must include accuracy and exactly one format family."},
    )
    max_pixels: Optional[int] = field(default=12845056, metadata={"help": "Maximum number of pixels for the image."})
    min_pixels: Optional[int] = field(default=3136, metadata={"help": "Minimum number of pixels for the image."})
    use_tool: bool = field(default=True, metadata={"help": "Whether to use tool-enabled generation."})
    tool_runtime_variant: str = field(default="safe", metadata={"help": "Tool runtime implementation to use: 'safe' or 'legacy'."})
    system_prompt_variant: str = field(default="auto", metadata={"help": "Instruction prompt style to use: 'auto', 'strict', or 'legacy'."})
    query_key: Optional[str] = field(default="question")
    controller_addr: Optional[str] = field(default="http://SH-IDCA1404-10-140-54-5:20001", metadata={"help": "Address of the tool controller."})
    use_paper_step_judge_reward: bool = field(default=True, metadata={"help": "Enable the paper-style step-judge trainer."})
    paper_step_judge_model: str = field(default="Qwen/Qwen2.5-VL-72B-Instruct", metadata={"help": "Judge model name."})
    paper_step_judge_temperature: float = field(default=0.0, metadata={"help": "Judge sampling temperature."})
    paper_step_judge_do_sample: bool = field(default=False, metadata={"help": "Whether to sample from the judge."})
    paper_step_judge_top_p: float = field(default=1.0, metadata={"help": "Judge top-p value when sampling."})
    paper_step_judge_top_k: int = field(default=0, metadata={"help": "Judge top-k value when sampling."})
    paper_step_judge_repetition_penalty: float = field(default=1.0, metadata={"help": "Judge repetition penalty."})
    paper_step_judge_max_new_tokens: int = field(default=128, metadata={"help": "Maximum new tokens for judge generation."})
    paper_terminal_weight: float = field(default=10.0, metadata={"help": "Weight applied to terminal accuracy reward."})
    paper_format_weight: float = field(default=0.5, metadata={"help": "Weight applied to the selected format reward."})
    paper_beta: float = field(default=0.5, metadata={"help": "Mixing factor between local turn reward and final reward."})
    paper_credit_assignment_mode: str = field(default="whole_turn", metadata={"help": "Token credit assignment mode: 'whole_turn' or 'boundary_token'."})
    paper_step_crash_filter: bool = field(default=True, metadata={"help": "Filter judge-crashed trajectories out of grouped stats and loss."})
    paper_log_judge_outputs: bool = field(default=True, metadata={"help": "Whether to print judge outputs during training."})


def _get_paper_step_judge_tool_vllm_trainer(runtime_variant: str):
    if runtime_variant == "safe":
        return Qwen2VLGRPOPaperStepJudgeVLLMTrainerSafe
    if runtime_variant == "legacy":
        return Qwen2VLGRPOPaperStepJudgeVLLMTrainerLegacy
    raise ValueError(
        f"Unsupported tool_runtime_variant={runtime_variant!r}. Expected one of: 'safe', 'legacy'."
    )


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    if not script_args.use_paper_step_judge_reward:
        raise ValueError("This entrypoint is dedicated to the paper-style step judge. Set --use_paper_step_judge_reward true.")
    if not script_args.use_tool or not training_args.use_vllm:
        raise ValueError("Paper step-judge reward requires --use_tool true and --use_vllm True.")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    system_prompt, resolved_prompt_variant = _get_system_prompt(
        script_args.system_prompt_variant, script_args.reward_funcs
    )
    dataset = load_dataset("json", data_files=script_args.dataset_name)

    def load_image_from_path(example):
        if "solution" not in example:
            example["solution"] = example["label"]
        if "label" in example:
            example.pop("label", None)
        image = Image.open(example["image_path"])
        example["image"] = image.convert("RGBA")
        return example

    question_template = "{Question}"

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
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

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example[script_args.query_key]},
            ],
        }

    if "image_path" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(load_image_from_path)
        dataset = dataset.map(make_conversation_image)
    else:
        dataset = dataset.map(make_conversation)
        if "query" in dataset[script_args.dataset_train_split].column_names:
            dataset = dataset.remove_columns("query")

    trainer_cls = _get_paper_step_judge_tool_vllm_trainer(script_args.tool_runtime_variant)
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
        reward_func_names=script_args.reward_funcs,
        paper_step_judge_model=script_args.paper_step_judge_model,
        paper_step_judge_temperature=script_args.paper_step_judge_temperature,
        paper_step_judge_do_sample=script_args.paper_step_judge_do_sample,
        paper_step_judge_top_p=script_args.paper_step_judge_top_p,
        paper_step_judge_top_k=script_args.paper_step_judge_top_k,
        paper_step_judge_repetition_penalty=script_args.paper_step_judge_repetition_penalty,
        paper_step_judge_max_new_tokens=script_args.paper_step_judge_max_new_tokens,
        paper_terminal_weight=script_args.paper_terminal_weight,
        paper_format_weight=script_args.paper_format_weight,
        paper_beta=script_args.paper_beta,
        paper_credit_assignment_mode=script_args.paper_credit_assignment_mode,
        paper_step_crash_filter=script_args.paper_step_crash_filter,
        paper_log_judge_outputs=script_args.paper_log_judge_outputs,
    )

    print("using:", trainer_cls)
    print("reward_funcs:", script_args.reward_funcs)
    print("system_prompt_variant:", resolved_prompt_variant)
    print("tool_runtime_variant:", script_args.tool_runtime_variant)
    print("paper_step_judge_model:", script_args.paper_step_judge_model)
    print("paper_credit_assignment_mode:", script_args.paper_credit_assignment_mode)
    print("paper_step_crash_filter:", script_args.paper_step_crash_filter)

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
    parser = TrlParser((PaperStepJudgeScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
