#!/usr/bin/env bash
set -euo pipefail

# -------- User-editable variables --------
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-3B-Instruct"   # Base vision-language model to fine-tune.
DATASET_JSON="/path/to/train.json"                 # JSON dataset path passed to --dataset_name.
OUTPUT_DIR="./outputs/tool_gfn_tb"                 # Directory where checkpoints/logs are written.
DEEPSPEED_CONFIG="./configs/ds_tool_gfn_tb_zero2.json"  # DeepSpeed ZeRO-2 config file.

FINETUNE_MODE="lora"             # lora = adapter tuning, full = full fine-tuning.
FREEZE_VISION_TOWER="true"       # Freeze vision encoder parameters to save VRAM.
MAX_ROUNDS=4                      # Maximum number of tool-calling turns per rollout.
NUM_GENERATIONS=2                 # Number of rollout samples generated per prompt.

REWARD_ACCURACY_WEIGHT=1.0        # Weight on final-answer accuracy reward.
REWARD_FORMAT_WEIGHT=0.25         # Weight on strict-format reward.
REWARD_EPSILON=1e-4               # Small positive floor added before taking log reward.

REPLAY_BUFFER_SIZE=2048           # Number of trajectories stored in replay.
REPLAY_SAMPLING="prioritized"    # prioritized or uniform replay sampling.
REPLAY_PRIORITY_ALPHA=1.0         # Priority exponent; 0.0 makes prioritized behave like uniform.
ROLLOUT_SYNC_INTERVAL=8           # Sync HF training weights into vLLM every N optimizer steps.

PER_DEVICE_TRAIN_BATCH_SIZE=1     # Micro-batch size per GPU for training.
GRADIENT_ACCUMULATION_STEPS=8     # Number of micro-batches accumulated before an optimizer step.
LEARNING_RATE=1e-5                # Main model learning rate.
LOGZ_LR=1e-2                      # Separate learning rate for scalar logZ.
MAX_PROMPT_LENGTH=2048            # Max prompt tokens for the vLLM/HF processor side.
MAX_COMPLETION_LENGTH=768         # Max generated assistant tokens across a rollout.
VLLM_DEVICE="auto"               # Dedicated vLLM device. auto = next visible GPU after training ranks.
VLLM_GPU_MEMORY_UTILIZATION=0.60  # Fraction of the vLLM GPU reserved for inference engine allocations.

NUM_PROCESSES=1                   # Number of training processes / GPUs for accelerate.
MIXED_PRECISION="bf16"           # bf16 recommended on Ampere/Hopper.
SEED=42                           # Random seed.
MASTER_PORT=29521                 # Distributed rendezvous port.
# ---------------------------------------

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --main_process_port "${MASTER_PORT}" \
  --use_deepspeed \
  --deepspeed_config_file "${DEEPSPEED_CONFIG}" \
  -m r1_v.open_r1.tool_gfn_tb \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset_name "${DATASET_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --finetune_mode "${FINETUNE_MODE}" \
  --freeze_vision_tower "${FREEZE_VISION_TOWER}" \
  --max_rounds "${MAX_ROUNDS}" \
  --num_generations "${NUM_GENERATIONS}" \
  --reward_accuracy_weight "${REWARD_ACCURACY_WEIGHT}" \
  --reward_format_weight "${REWARD_FORMAT_WEIGHT}" \
  --reward_epsilon "${REWARD_EPSILON}" \
  --replay_buffer_size "${REPLAY_BUFFER_SIZE}" \
  --replay_sampling "${REPLAY_SAMPLING}" \
  --replay_priority_alpha "${REPLAY_PRIORITY_ALPHA}" \
  --rollout_sync_interval "${ROLLOUT_SYNC_INTERVAL}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --logZ_lr "${LOGZ_LR}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --vllm_device "${VLLM_DEVICE}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --use_vllm true \
  --bf16 true \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_steps 100 \
  --save_total_limit 3 \
  --num_train_epochs 1 \
  --report_to wandb \
  --seed "${SEED}"
