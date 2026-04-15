from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import broadcast_object_list, gather_object, set_seed
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from .tool_tbgfn_args import ToolTBGFNArgs
from .tool_tbgfn_loss import ToolMacroTrajectoryBalanceLoss
from .tool_tbgfn_policy import AssistantTurnLogProbScorer
from .tool_tbgfn_reward_adapter import ToolTBGFNRewardAdapter
from .tool_tbgfn_rollout import SafeToolTBGFNRolloutCollector

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover
    LLM = None
    SamplingParams = None

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


class ToolTBGFNTrainer:
    def __init__(self, args: ToolTBGFNArgs):
        self.args = args
        ds_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed else None
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            deepspeed_plugin=ds_plugin,
        )
        set_seed(args.seed)
        self._setup_wandb()
        self.dataset = self._build_dataset()
        self.dataloader = self._build_dataloader()
        self.model, self.processor = self._build_model_and_processor()
        self.loss_module = ToolMacroTrajectoryBalanceLoss(log_reward_clip_min=args.log_reward_clip_min)
        self.reward_adapter = ToolTBGFNRewardAdapter(
            accuracy_reward_fn_path=args.accuracy_reward_fn_path,
            format_reward_fn_path=args.format_reward_fn_path,
            log_reward_epsilon=args.log_reward_epsilon,
        )
        self.policy_scorer = AssistantTurnLogProbScorer(
            model=self.model,
            processor=self.processor,
            max_prompt_length=args.max_prompt_length,
        )
        self.optimizer = self._build_optimizer()
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )
        self.rollout_llm = self._build_rollout_llm()
        self.rollout_collector = SafeToolTBGFNRolloutCollector(
            controller_addr=args.controller_addr,
            max_rounds=args.max_rounds,
            sampling_params=self._build_sampling_params(),
        )
        self.global_step = 0

    def _setup_wandb(self) -> None:
        if self.args.report_to != "wandb" or wandb is None or not self.accelerator.is_main_process:
            self.wandb_run = None
            return
        self.wandb_run = wandb.init(
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            name=self.args.run_name,
            config=asdict(self.args),
        )

    def _build_dataset(self):
        return load_dataset(self.args.dataset_name, split=self.args.dataset_split)

    def _build_dataloader(self):
        def collate_fn(features: List[Dict[str, Any]]):
            return features
        return DataLoader(
            self.dataset,
            batch_size=self.args.per_device_prompt_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=collate_fn,
        )

    def _build_model_and_processor(self):
        model_name = self.args.model_name_or_path
        processor = AutoProcessor.from_pretrained(model_name)
        kwargs = {
            "torch_dtype": torch.bfloat16 if self.args.bf16 else torch.float16,
            "attn_implementation": self.args.attn_implementation,
        }
        if "Qwen2.5-VL" in model_name:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
        elif "Qwen2-VL" in model_name or "Qwen2VL" in model_name:
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model, processor

    def _build_optimizer(self):
        optim = torch.optim.AdamW(
            [
                {"params": [p for p in self.model.parameters() if p.requires_grad], "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay},
                {"params": self.loss_module.parameters(), "lr": self.args.logz_learning_rate, "weight_decay": 0.0},
            ],
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return optim

    def _build_rollout_llm(self):
        if not self.args.use_vllm:
            return None
        if LLM is None:
            raise ImportError("vllm is required for rollout when --use_vllm true")
        if not self.accelerator.is_main_process:
            return None
        device = self.args.vllm_device
        if device == "auto":
            device = f"cuda:{self.accelerator.num_processes}"
        return LLM(
            model=self.args.model_name_or_path,
            device=device,
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            dtype=torch.bfloat16 if self.args.bf16 else torch.float16,
            enable_prefix_caching=True,
            enforce_eager=True,
        )

    def _build_sampling_params(self):
        if SamplingParams is None:
            return None
        return SamplingParams(
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.max_completion_length,
        )

    def _load_image(self, image_value: Any):
        if isinstance(image_value, Image.Image):
            return image_value
        if isinstance(image_value, str):
            return Image.open(image_value).convert("RGB")
        if isinstance(image_value, dict) and "path" in image_value:
            return Image.open(image_value["path"]).convert("RGB")
        raise TypeError(f"Unsupported image value type: {type(image_value)}")

    def _extract_batch_prompts_and_images(self, batch: List[Dict[str, Any]]):
        prompts = []
        images = []
        for example in batch:
            prompts.append(example[self.args.prompt_column])
            images.append(self._load_image(example[self.args.image_column]))
        return prompts, images

    def _broadcast_traces(self, local_prompts: List[Any], local_images: List[Any]):
        all_prompts = gather_object(local_prompts)
        all_images = gather_object(local_images)
        local_num_traces = len(local_prompts) * self.args.num_trajectories_per_prompt
        if self.accelerator.is_main_process:
            traces, prompt_indices, traj_indices = self.rollout_collector.collect(
                self.rollout_llm,
                all_prompts,
                all_images,
                self.args.num_trajectories_per_prompt,
            )
            payload = []
            for trace, prompt_index, traj_index in zip(traces, prompt_indices, traj_indices):
                payload.append({
                    "trace": trace,
                    "prompt_index": prompt_index,
                    "traj_index": traj_index,
                })
        else:
            payload = [None] * (len(all_prompts) * self.args.num_trajectories_per_prompt)
        payload = broadcast_object_list(payload, from_process=0)
        start = self.accelerator.process_index * local_num_traces
        end = start + local_num_traces
        return payload[start:end]

    def _compute_local_rewards_and_logpf(self, payload_slice: List[dict]):
        traces = [item["trace"] for item in payload_slice]
        raw_rewards, log_rewards = self.reward_adapter.compute_log_rewards(traces)
        trajectory_log_pf = torch.stack([self.policy_scorer.score_trace(trace) for trace in traces]).to(self.model.device)
        log_rewards = torch.tensor(log_rewards, dtype=trajectory_log_pf.dtype, device=trajectory_log_pf.device)
        raw_rewards_tensor = torch.tensor(raw_rewards, dtype=trajectory_log_pf.dtype, device=trajectory_log_pf.device)
        return raw_rewards_tensor, log_rewards, trajectory_log_pf

    def _log_metrics(self, raw_rewards: torch.Tensor):
        gathered = self.accelerator.gather(raw_rewards.detach())
        if self.accelerator.is_main_process:
            avg_reward = gathered.mean().item()
            reward_variance = gathered.var(unbiased=False).item()
            if self.wandb_run is not None:
                wandb.log(
                    {
                        "train/avg_reward_per_trajectory": avg_reward,
                        "train/reward_variance": reward_variance,
                        "train/global_step": self.global_step,
                    },
                    step=self.global_step,
                )
            else:
                print({
                    "train/avg_reward_per_trajectory": avg_reward,
                    "train/reward_variance": reward_variance,
                    "train/global_step": self.global_step,
                })

    def _save_checkpoint(self):
        if not self.accelerator.is_main_process:
            return
        output_dir = Path(self.args.output_dir) / f"checkpoint-{self.global_step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped.state_dict(), output_dir / "pytorch_model.bin")
        self.processor.save_pretrained(output_dir)
        torch.save(self.loss_module.state_dict(), output_dir / "tb_objective.bin")
        with open(output_dir / "trainer_state.json", "w", encoding="utf-8") as f:
            json.dump({"global_step": self.global_step}, f)

    def train(self):
        self.model.train()
        for epoch in range(self.args.num_train_epochs):
            for batch in self.dataloader:
                with self.accelerator.accumulate(self.model):
                    local_prompts, local_images = self._extract_batch_prompts_and_images(batch)
                    payload_slice = self._broadcast_traces(local_prompts, local_images)
                    raw_rewards, log_rewards, trajectory_log_pf = self._compute_local_rewards_and_logpf(payload_slice)
                    loss = self.loss_module(trajectory_log_pf=trajectory_log_pf, log_rewards=log_rewards)
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                self.global_step += 1
                if self.global_step % self.args.logging_steps == 0:
                    self._log_metrics(raw_rewards)
                if self.global_step % self.args.save_steps == 0:
                    self._save_checkpoint()
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    if self.accelerator.is_main_process and self.wandb_run is not None:
                        self.wandb_run.finish()
                    return
        if self.accelerator.is_main_process and self.wandb_run is not None:
            self.wandb_run.finish()
