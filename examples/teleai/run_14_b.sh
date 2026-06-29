#!/bin/bash
export ORION_GMEM_CONTROL=v1
# Run model
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH=$PYTHONPATH:/gemini/space/yifq/yifq/code/Megatron-LM
export PYTHONPATH=$PYTHONPATH:/gemini/space/yifq/yifq/code/teleai_data_tool_source_code
export CPLUS_INCLUDE_PATH=/usr/include/python3.10:$CPLUS_INCLUDE_PATH
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/qiuyang/Video-Depth-Anything/
####################################### IMPORTANT ARGS, DO NOT MODIFY #######################################
# Parallel config 
CP=1 # 12 % CP == 0
TP=1 # not support

# assert num_tensor_and_context_groups(N_GPU_FOR_TRAIN / CP) % producer_size(N_GPU_FOR_DATA) == 0 and num_tensor_and_context_groups//producer_size >=1
# Multi-node config 
# N_MOE=1
# N_GPU_FOR_TRAIN=24  # 3:1 ratio
# N_GPU_FOR_DATA=8

# Single-node config 
N_MOE=1
N_GPU_FOR_TRAIN=6
N_GPU_FOR_DATA=2

EXPR_NAME=fl2v_14B_recon_multi_resolution_f45_sft

TRAIN_SCRIPT=${1:-"examples/teleai/pretrain_s2v_recon.py"}
CONFIG_PATH=${2:-"config.teleai_i2v_multiresolution_14b.config"}
shift
echo "Launching: $TRAIN_SCRIPT"


TENSORBOARD_LOGS_PATH=./logs/${EXPR_NAME}
CHECKPOINT_PATH_LOAD=/gemini/space/yifq/teletron-model/Wan2.1-FLF2V-14B-720P
CHECKPOINT_PATH_SAVE=/gemini/space/yanjq/Teletron/Teletron_1_3b_v1/workdirs/${EXPR_NAME} # use your folder
####################################### IMPORTANT ARGS END #######################################

mkdir -p $CHECKPOINT_PATH_SAVE

MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
MASTER_PORT='11322'
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
    --train-iters 200000
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
    --save $CHECKPOINT_PATH_SAVE
    
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
    --save-interval 500
    --eval-interval 2000000
    # --load $CHECKPOINT_PATH_LOAD 
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
    "$@"
