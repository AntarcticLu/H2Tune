#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
port=${1}
dataset=${2}
model_path=${3}
round=${4}
epoch=${5}
batch=${6}
lora_r=${7}
max_r=${8}
max_layer=${9}
strategy=${10}
learn_rate=${11}
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
torchrun \
--nproc_per_node $GPUS_PER_NODE \
--nnodes $NNODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port ${port} \
./code/finetune.py \
--model_name_or_path ${model_path} \
--data_path ${dataset} \
--fp16 True \
--output_dir ./savedir/llm_${port} \
--num_train_epochs ${epoch} \
--per_device_train_batch_size ${batch} \
--gradient_accumulation_steps 8 \
--save_strategy "steps" \
--save_steps 5 \
--save_total_limit 1 \
--learning_rate ${learn_rate} \
--weight_decay 0.1 \
--adam_beta2 0.95 \
--warmup_ratio 0.01 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "none" \
--model_max_length 512 \
--gradient_checkpointing False \
--deepspeed "./code/ds_config_zero2.json" \
--use_lora True \
--round ${round} \
--strategy ${strategy} \
--max_r ${max_r} \
--lora_r ${lora_r} \
--max_layer ${max_layer}
