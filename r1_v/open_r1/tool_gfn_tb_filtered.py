from r1_v.open_r1.tool_gfn_tb import GFNTBScriptArguments, STRICT_SYSTEM_PROMPT
from r1_v.open_r1.trainer.tool_vllm_gfn_tb_trainer_safe_filtered import Qwen2VLGFNTBVLLMTrainerFiltered

import os
from datasets import load_dataset
from PIL import Image
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, ModelConfig, TrlParser, get_peft_config


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

    trainer = Qwen2VLGFNTBVLLMTrainerFiltered(
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
