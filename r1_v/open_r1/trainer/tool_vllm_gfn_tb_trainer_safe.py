import math
import uuid
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Optional, Union
from unittest.mock import patch
import warnings

import torch
from accelerate.utils import broadcast_object_list, gather_object
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from qwen_vl_utils import process_vision_info
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available

from trl.import_utils import is_vllm_available
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

from .tool_generation_safe_gfn import vllm_generate_with_tool_calls_gfn
from .tool_replay_buffer import ReplayItem, ToolTrajectoryReplayBuffer
from .tool_tb_rewards import accuracy_reward_from_model_outputs, format_reward_from_model_outputs

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb


class Qwen2VLGFNTBVLLMTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers=(None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        controller_addr: str = "http://SH-IDCA1404-10-140-54-5:20001",
        max_rounds: int = 4,
        reward_accuracy_weight: float = 1.0,
        reward_format_weight: float = 0.25,
        reward_epsilon: float = 1e-4,
        replay_buffer_size: int = 2048,
        replay_sampling: str = "prioritized",
        replay_priority_alpha: float = 1.0,
        rollout_sync_interval: int = 8,
        freeze_vision_tower: bool = True,
        log_reward_clip_min: float = -9.21,
        logZ_init: float = 0.0,
        logZ_lr: float = 1e-2,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GFN-TB")

        self.controller_addr = controller_addr
        self.max_rounds = max_rounds
        self.reward_accuracy_weight = reward_accuracy_weight
        self.reward_format_weight = reward_format_weight
        self.reward_epsilon = reward_epsilon
        self.replay_sampling = replay_sampling
        self.replay_priority_alpha = replay_priority_alpha
        self.rollout_sync_interval = max(1, rollout_sync_interval)
        self.log_reward_clip_min = log_reward_clip_min
        self.logZ_lr = logZ_lr
        self.replay_buffer = ToolTrajectoryReplayBuffer(
            capacity=replay_buffer_size,
            sampling_mode=replay_sampling,
            priority_alpha=replay_priority_alpha,
            seed=args.seed,
        )
        self._pending_logZ_init = float(logZ_init)
        self._metrics = defaultdict(list)

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        model_init_kwargs["torch_dtype"] = torch.float16

        if isinstance(model, str):
            model_id = model
            model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            if "Qwen2-VL" in model_id or "Qwen2VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache", None)
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        if freeze_vision_tower:
            self._freeze_vision_tower(model)

        if not hasattr(model, "gfn_logZ"):
            model.register_parameter("gfn_logZ", torch.nn.Parameter(torch.tensor(self._pending_logZ_init, dtype=torch.float32)))
        self.logZ = model.gfn_logZ

        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                processing_class = AutoTokenizer.from_pretrained(model_id, padding_side="left")

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.use_vllm = args.use_vllm
        self.model_id = model_id

        def data_collator(features):
            return features

        if hasattr(model, "warnings_issued"):
            model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        self.model_accepts_loss_kwargs = False

        if not self.use_vllm:
            raise ValueError("Qwen2VLGFNTBVLLMTrainer requires --use_vllm true")
        if not is_vllm_available():
            raise ImportError("vLLM is required for Qwen2VLGFNTBVLLMTrainer")

        if self.accelerator.is_main_process:
            vllm_device = self.args.vllm_device
            if vllm_device == "auto":
                vllm_device = f"cuda:{self.accelerator.num_processes}"
            if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                raise ValueError(f"Requested vLLM device {vllm_device} is not available")
            if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                warnings.warn(f"The requested device {vllm_device} is also used for training.")
            world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
            profiling_patch = patch(
                "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                return_value=None,
            )
            with world_size_patch, profiling_patch:
                self.llm = LLM(
                    model=model_id,
                    device=vllm_device,
                    gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                    dtype=torch.bfloat16,
                    enable_prefix_caching=True,
                    enforce_eager=True,
                    mm_processor_kwargs=(
                        {"max_pixels": max_pixels, "min_pixels": min_pixels}
                        if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id
                        else None
                    ),
                    max_model_len=args.max_prompt_length,
                    limit_mm_per_prompt={"image": 6},
                )
            self.sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=self.max_completion_length,
            )
        self._last_loaded_step = -1
        self.accelerator.wait_for_everyone()

    def _freeze_vision_tower(self, model: PreTrainedModel) -> None:
        for name, param in model.named_parameters():
            lowered = name.lower()
            if "vision" in lowered or "visual" in lowered or "image" in lowered:
                param.requires_grad = False

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        decay_parameters = self.get_decay_parameter_names(self.model)
        named_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in named_params if n in decay_parameters and n != "gfn_logZ"],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in named_params if n not in decay_parameters and n != "gfn_logZ"],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in named_params if n == "gfn_logZ"],
                "weight_decay": 0.0,
                "lr": self.logZ_lr,
            },
        ]
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _maybe_sync_vllm_weights(self):
        if self.state.global_step == self._last_loaded_step:
            return
        if self.state.global_step > 0 and self.state.global_step % self.rollout_sync_interval != 0 and self._last_loaded_step >= 0:
            return

        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            gather_deepspeed3_params=False,
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                state_dict = unwrapped_model._orig_mod.state_dict()
            else:
                state_dict = unwrapped_model.state_dict()
        if self.accelerator.is_main_process:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(state_dict.items())
        self._last_loaded_step = self.state.global_step
        self.accelerator.wait_for_everyone()

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        prompts = [x["prompt"] for x in inputs]
        images = [x["image"] for x in inputs]
        solutions = [x.get("solution", "") for x in inputs]

        expanded_prompts = []
        expanded_images = []
        expanded_solutions = []
        for prompt, image, solution in zip(prompts, images, solutions):
            for _ in range(self.num_generations):
                expanded_prompts.append(prompt)
                expanded_images.append(image)
                expanded_solutions.append(solution)

        self._maybe_sync_vllm_weights()

        all_prompts = gather_object(expanded_prompts)
        all_images = gather_object(expanded_images)
        all_solutions = gather_object(expanded_solutions)

        if self.accelerator.is_main_process:
            rollout_outputs = vllm_generate_with_tool_calls_gfn(
                self.llm,
                prompts=all_prompts,
                images=all_images,
                sampling_params=self.sampling_params,
                max_rounds=self.max_rounds,
                model_mode="general",
                controller_addr=self.controller_addr,
            )
            replay_items = []
            for output_item, solution in zip(rollout_outputs, all_solutions):
                acc = accuracy_reward_from_model_outputs(output_item.get("model_outputs", []), solution)
                fmt = format_reward_from_model_outputs(output_item.get("model_outputs", []))
                reward_total = self.reward_epsilon + self.reward_accuracy_weight * acc + self.reward_format_weight * fmt
                log_reward = math.log(max(reward_total, self.reward_epsilon))
                if math.isfinite(self.log_reward_clip_min):
                    log_reward = max(log_reward, self.log_reward_clip_min)
                replay_items.append(
                    ReplayItem(
                        sample_id=str(uuid.uuid4()),
                        solution=solution,
                        turn_records=output_item.get("turn_records", []),
                        reward_accuracy=float(acc),
                        reward_format=float(fmt),
                        reward_total=float(reward_total),
                        log_reward=float(log_reward),
                        num_turns=len(output_item.get("turn_records", [])),
                    )
                )
            self.replay_buffer.extend(replay_items)
            sampled = self.replay_buffer.sample(len(all_prompts)) or replay_items
            ave_reward = sum(item.reward_total for item in sampled) / max(len(sampled), 1)
            ave_acc = sum(item.reward_accuracy for item in sampled) / max(len(sampled), 1)
            ave_fmt = sum(item.reward_format for item in sampled) / max(len(sampled), 1)
            ave_turns = sum(item.num_turns for item in sampled) / max(len(sampled), 1)
            self._metrics["reward_total"].append(ave_reward)
            self._metrics["reward_accuracy"].append(ave_acc)
            self._metrics["reward_format"].append(ave_fmt)
            self._metrics["num_turns"].append(ave_turns)
            self._metrics["replay_size"].append(float(len(self.replay_buffer)))
        else:
            sampled = [None] * len(all_prompts)

        sampled = broadcast_object_list(sampled, from_process=0)
        local_n = len(expanded_prompts)
        process_slice = slice(
            self.accelerator.process_index * local_n,
            (self.accelerator.process_index + 1) * local_n,
        )
        local_replay_items = sampled[process_slice]
        return {"replay_items": local_replay_items}

    def _score_turn_logprob(self, model: PreTrainedModel, turn_record: dict[str, Any]) -> torch.Tensor:
        state_messages = turn_record["state_messages_before_turn"]
        assistant_action_text = turn_record["assistant_action_text"]
        full_messages = [*state_messages, {"role": "assistant", "content": [{"type": "text", "text": assistant_action_text}]}]

        prefix_text = self.processing_class.apply_chat_template(
            state_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.processing_class.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prefix_images, _ = process_vision_info(state_messages)
        full_images, _ = process_vision_info(full_messages)

        prefix_inputs = self.processing_class(
            text=[prefix_text],
            images=prefix_images,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        full_inputs = self.processing_class(
            text=[full_text],
            images=full_images,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        prefix_len = prefix_inputs["input_ids"].shape[1]
        input_ids = full_inputs["input_ids"].to(model.device)
        attention_mask = full_inputs["attention_mask"].to(model.device)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "pixel_values" in full_inputs:
            model_inputs["pixel_values"] = full_inputs["pixel_values"].to(model.device, dtype=torch.bfloat16)
        if "image_grid_thw" in full_inputs:
            model_inputs["image_grid_thw"] = full_inputs["image_grid_thw"].to(model.device)

        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if model.device.type == "cuda" else nullcontext()
        with autocast_ctx:
            logits = model(**model_inputs).logits

        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        start_idx = max(prefix_len - 1, 0)
        logits = logits[:, start_idx:, :]
        targets = targets[:, start_idx:]
        log_probs = logits.log_softmax(dim=-1)
        token_logps = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        return token_logps.sum()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("Qwen2VLGFNTBVLLMTrainer does not support return_outputs")
        replay_items = inputs["replay_items"]
        losses = []
        log_pf_values = []
        for item in replay_items:
            if item is None:
                continue
            log_pf = torch.zeros((), device=model.device, dtype=torch.float32)
            for turn_record in item.turn_records:
                log_pf = log_pf + self._score_turn_logprob(model, turn_record).float()
            residual = self.logZ + log_pf - torch.tensor(item.log_reward, device=model.device, dtype=torch.float32)
            losses.append(residual.pow(2))
            log_pf_values.append(log_pf.detach())

        if not losses:
            loss = self.logZ * 0.0
        else:
            loss = torch.stack(losses).mean()
            mean_log_pf = torch.stack(log_pf_values).mean()
            self._metrics["log_pf"].append(mean_log_pf.item())
            self._metrics["logZ"].append(self.logZ.detach().float().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        if logs and next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics.clear()
