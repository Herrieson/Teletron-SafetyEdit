#!/bin/bash
export ORION_GMEM_CONTROL=v1
# Run model
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH=$PYTHONPATH:/gemini/platform/shared/yifq1/Megatron-LM
export PYTHONPATH=$PYTHONPATH:/gemini/platform/shared/yifq1/teleai_data_tool_source_code
export CPLUS_INCLUDE_PATH=/usr/include/python3.10:$CPLUS_INCLUDE_PATH
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/qiuyang/Video-Depth-Anything/
####################################### IMPORTANT ARGS, DO NOT MODIFY #######################################
# Parallel config 
CP=1 # 12 % CP == 0
TP=1 # not support

# assert num_tensor_and_context_groups(N_GPU_FOR_TRAIN / CP) % producer_size(N_GPU_FOR_DATA) == 0 and num_tensor_and_context_groups//producer_size >=1
# Multi-node config 
N_MOE=1
N_GPU_FOR_TRAIN=6  # 3:1 ratio
N_GPU_FOR_DATA=2

# Single-node config 
# N_MOE=1
# N_GPU_FOR_TRAIN=4
# N_GPU_FOR_DATA=4

EXPR_NAME=fl2v_1.3B_recon_multi_resolution_f29_sft_randomcanny_tae

TRAIN_SCRIPT=${1:-"examples/teleai/pretrain_s2v_recon_tae_ds.py"}
CONFIG_PATH=${2:-"config.teleai_i2v_multiresolution_v1_dynamic_ds.config"}
shift
echo "Launching: $TRAIN_SCRIPT"

TENSORBOARD_LOGS_PATH=./logs/${EXPR_NAME}
CHECKPOINT_PATH_LOAD=/gemini/user/shared/workdirs/fl2v_1.3B_recon_multi_resolution_f29_sft_randomcanny_tae_lowresolution_encaug
CHECKPOINT_PATH_SAVE=/gemini/user/shared/workdirs/fl2v_1.3B_recon_multi_resolution_f29_sft_randomcanny_tae_lowresolution_encaug_disableattn # use your folder

# CHECKPOINT_PATH_LOAD_DISC=/gemini/space/yifq/teletron-model/Wan2.1-Fun-V1.1-1.3B-InP
####################################### IMPORTANT ARGS END #######################################

mkdir -p $CHECKPOINT_PATH_SAVE

MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
MASTER_PORT='11321'
NODE_RANK=${RANK:-'0'}

MBS=1
N_GPU=$((N_GPU_FOR_TRAIN+N_GPU_FOR_DATA))
NNODES=$((($N_GPU-1)/8+1))
WORLD_SIZE=$N_GPU_FOR_TRAIN

N_VAE=$N_GPU_FOR_DATA
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

if [ $NNODES -eq 1 ]; then
    N_PROC=$N_GPU
else
    N_PROC=8
fi

echo '$MASTER_ADDR' $MASTER_ADDR
echo '$NODE_RANK & $NNODES' $NODE_RANK $NNODES
echo '$N_GPU_FOR_TRAIN' $N_GPU_FOR_TRAIN
echo '$N_GPU_FOR_DATA' $N_GPU_FOR_DATA

DISTRIBUTED_ARGS=(
    --nproc_per_node $N_PROC 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)


TRAINING_ARGS=( 
    --micro-batch-size ${MBS}
    --train-iters 40000
    --weight-decay 1e-4
    --init-method-std 0.006 
    --clip-grad 1.0
    --bf16
    --lr 1e-5
    --lr-decay-style constant
    --lr-warmup-fraction 0
    --recompute-granularity full 
    --recompute-method block 
    # --activation-offload
    --use-distributed-optimizer
    --recompute-num-layers 40
    --no-rope-fusion
    --distributed-timeout-minutes 60
    --override-opt_param-scheduler
    --data-parallel-random-init
    # --pretrained_model_path $CHECKPOINT_PATH_LOAD 
    --load $CHECKPOINT_PATH_LOAD 
    # --load_disc $CHECKPOINT_PATH_LOAD_DISC
    --save $CHECKPOINT_PATH_SAVE

    --finetune
    --no-load-rng
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
    --distributed-vae
    --distributed-vae-world-size $N_VAE
    --consumer-models-num $N_MOE
)
DATA_ARGS=(
    --split 949,50,1
    --num-workers 1
    --config-path ${CONFIG_PATH}
)

EVAL_AND_LOGGING_ARGS=(
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 1
    --tensorboard-queue-size 10
    --log-interval 1 # for terminal infos
    --save-interval 2000
    --eval-interval 2000000
    #--load $CHECKPOINT_PATH_LOAD 
    # --save $CHECKPOINT_PATH_SAVE
    --eval-iters 20 # sample 20 video to eval
    --producer-log-level 2 # 1: debug | 2: Info
)

torchrun ${DISTRIBUTED_ARGS[@]} ${TRAIN_SCRIPT} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${LORA_CFG[@]} \
    "$@" \
    2>&1 | tee -a ./logs/fl2v_1.3B_recon_multi_resolution_f77_sft_w_canny.log

