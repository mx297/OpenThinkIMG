#!/usr/bin/env bash

set -e
export PYTHONPATH="$(pwd)"

# =========================
# GPU / CUDA
# =========================
export CUDA_VISIBLE_DEVICES=0,3          # training GPU, vLLM GPU
export VLLM_LOGGING_LEVEL=ERROR          # reduce vLLM log verbosity

GPUS_PER_NODE=1                          # number of training processes per node
NNODES=1                                 # number of nodes
NODE_RANK=0                              # rank of this node
MASTER_ADDR=localhost                    # torch.distributed master address
MASTER_PORT=6007                         # torch.distributed master port

DISTRIBUTED_ARGS="
  --nproc_per_node ${GPUS_PER_NODE} \
  --nnodes ${NNODES} \
  --node_rank ${NODE_RANK} \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT}
"

export NCCL_P2P_DISABLE=1                # disable NCCL P2P if needed for multi-GPU stability
export NCCL_SHM_DISABLE=0                # keep NCCL shared-memory transport enabled

# =========================
# W&B
# =========================
export WANDB_ENTITY="matef297-mbzuai"   # Weights & Biases entity
export WANDB_PROJECT="OpenThinkIMG-GFlowNet"  # Weights & Biases project

# =========================
# Paths
# =========================
output_dir="output/qwen2_5_vl_sft_grpo_paper_step_judge"   # output directory
model_path="/share_5/users/mohamed_atef/OpenThinkIMG/output/Qwen2.5-VL_sft/"  # SFT checkpoint path
data_path="/share_5/users/mohamed_atef/OpenThinkIMG/tool_dataset/records.jsonl" # training data path
RUN_NAME="qwen2_5_vl_sft_grpo_paper_step_judge"            # run name

# =========================
# Reward / prompt / runtime
# =========================
REWARD_FUNCS=("accuracy" "format")     # reward functions: must include accuracy and one format family
SYSTEM_PROMPT_VARIANT="strict"          # one of: auto, strict, legacy
TOOL_RUNTIME_VARIANT="safe"             # one of: safe, legacy
USE_TOOL="true"                         # enable tool-enabled generation
USE_PAPER_STEP_JUDGE_REWARD="true"      # enable the paper-style step judge trainer

# =========================
# Paper step judge config
# =========================
PAPER_STEP_JUDGE_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"  # judge model name
PAPER_STEP_JUDGE_TEMPERATURE="0.0"     # judge temperature
PAPER_STEP_JUDGE_DO_SAMPLE="false"     # whether the judge samples
PAPER_STEP_JUDGE_TOP_P="1.0"           # judge top-p when sampling
PAPER_STEP_JUDGE_TOP_K="0"             # judge top-k when sampling
PAPER_STEP_JUDGE_REPETITION_PENALTY="1.0"  # judge repetition penalty
PAPER_STEP_JUDGE_MAX_NEW_TOKENS="128"  # max new tokens for judge output
PAPER_TERMINAL_WEIGHT="10.0"           # weight for final accuracy reward
PAPER_FORMAT_WEIGHT="0.5"              # weight for the separate format reward in final reward
PAPER_BETA="0.5"                       # mixing factor between local turn reward and final reward
PAPER_CREDIT_ASSIGNMENT_MODE="whole_turn"  # one of: whole_turn, boundary_token
PAPER_STEP_CRASH_FILTER="true"         # if true, filter judge-crashed trajectories from grouped stats and loss
PAPER_LOG_JUDGE_OUTPUTS="true"         # print judge outputs during training

# =========================
# Core training hyperparameters
# =========================
LEARNING_RATE="1e-6"                   # optimizer learning rate
LR_SCHEDULER_TYPE="constant"           # scheduler type, e.g. constant, cosine, linear
NUM_GENERATIONS="8"                    # number of sampled trajectories per prompt
PER_DEVICE_TRAIN_BATCH_SIZE="8"        # batch size per device
GRADIENT_ACCUMULATION_STEPS="16"       # gradient accumulation steps
MAX_PROMPT_LENGTH="16000"              # max prompt tokens
MAX_COMPLETION_LENGTH="2048"           # max generated completion tokens
TEMPERATURE="1.0"                      # sampling temperature for the policy model
SEED="42"                              # random seed
VLLM_GPU_MEMORY_UTILIZATION="0.8"      # fraction of vLLM GPU memory to use
BF16="true"                            # enable bfloat16 training
GRADIENT_CHECKPOINTING="true"          # enable gradient checkpointing
MAX_PIXELS="200000"                    # max image pixels passed to processor
NUM_TRAIN_EPOCHS="1"                   # number of training epochs
SAVE_STEPS="100"                       # checkpoint save interval
LOGGING_STEPS="1"                      # log interval
CONTROLLER_ADDR="http://localhost:20101"   # tool controller endpoint

# =========================
# Launch
# =========================
torchrun ${DISTRIBUTED_ARGS} \
  r1_v/open_r1/tool_grpo_step_judge_paper.py \
  --use_vllm True \
  --output_dir "${output_dir}" \
  --model_name_or_path "${model_path}" \
  --dataset_name "${data_path}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --temperature "${TEMPERATURE}" \
  --seed "${SEED}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_generations "${NUM_GENERATIONS}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --bf16 "${BF16}" \
  --report_to wandb \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --max_pixels "${MAX_PIXELS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --run_name "${RUN_NAME}" \
  --save_steps "${SAVE_STEPS}" \
  --save_only_model true \
  --controller_addr "${CONTROLLER_ADDR}" \
  --use_tool "${USE_TOOL}" \
  --tool_runtime_variant "${TOOL_RUNTIME_VARIANT}" \
  --system_prompt_variant "${SYSTEM_PROMPT_VARIANT}" \
  --reward_funcs "${REWARD_FUNCS[@]}" \
  --use_paper_step_judge_reward "${USE_PAPER_STEP_JUDGE_REWARD}" \
  --paper_step_judge_model "${PAPER_STEP_JUDGE_MODEL}" \
  --paper_step_judge_temperature "${PAPER_STEP_JUDGE_TEMPERATURE}" \
  --paper_step_judge_do_sample "${PAPER_STEP_JUDGE_DO_SAMPLE}" \
  --paper_step_judge_top_p "${PAPER_STEP_JUDGE_TOP_P}" \
  --paper_step_judge_top_k "${PAPER_STEP_JUDGE_TOP_K}" \
  --paper_step_judge_repetition_penalty "${PAPER_STEP_JUDGE_REPETITION_PENALTY}" \
  --paper_step_judge_max_new_tokens "${PAPER_STEP_JUDGE_MAX_NEW_TOKENS}" \
  --paper_terminal_weight "${PAPER_TERMINAL_WEIGHT}" \
  --paper_format_weight "${PAPER_FORMAT_WEIGHT}" \
  --paper_beta "${PAPER_BETA}" \
  --paper_credit_assignment_mode "${PAPER_CREDIT_ASSIGNMENT_MODE}" \
  --paper_step_crash_filter "${PAPER_STEP_CRASH_FILTER}" \
  --paper_log_judge_outputs "${PAPER_LOG_JUDGE_OUTPUTS}"
