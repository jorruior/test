#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/training_%j.out
#SBATCH --time=14-00:00:00
#SBATCH --mem=1200G
#SBATCH --gres=gpu:a40:8
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=3
#SBATCH --partition=gpu
#SBATCH --export=ALL

# Trains RiboTransPred with DDP across 3 nodes / 24 GPUs

# ============ ENVIRONMENT ============
source ~/.bashrc
mamba activate ribotranspred

# ============ NCCL (InfiniBand) ============
export NCCL_SOCKET_IFNAME=ibp23s0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_SL=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=SYS
export NCCL_BUFFSIZE=2097152
export NCCL_NTHREADS=64
export NCCL_SOCKET_NTHREADS=8
export NCCL_CHECKS_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_TIMEOUT=1800

export GLOO_SOCKET_IFNAME=ibp23s0

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600

# ============ DISTRIBUTED ============
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + RANDOM % 45000))

# ============ DEBUG ============
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# ============ TRAINING CONFIGURATION ============
TRACKS=${1:?"Usage: sbatch train.sh <tracks_file>"}
REGION_LEN=6000
NBINS=1000
MODEL="PosTransModelTCN"
BIOTYPE="protein_coding"

BATCH_SIZE=4
MAX_EPOCHS=80
NHEADS=6
DROPOUT=0.3
LR=1e-5
WEIGHT_DECAY=0.0005
WARMUP=2000
GRAD_ACCUM=3
GRAD_CLIP=0.5

# ============ DIAGNOSTICS ============
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "========================================="
echo "Node list: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "========================================="
echo "DDP Configuration:"
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "========================================="
echo "Training Configuration:"
echo "REGION_LEN=$REGION_LEN"
echo "NBINS=$NBINS"
echo "MODEL=$MODEL"
echo "BIOTYPE=$BIOTYPE"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MAX_EPOCHS=$MAX_EPOCHS"
echo "NHEADS=$NHEADS"
echo "DROPOUT=$DROPOUT"
echo "LR=$LR"
echo "WEIGHT_DECAY=$WEIGHT_DECAY"
echo "WARMUP=$WARMUP"
echo "GRAD_ACCUM=$GRAD_ACCUM"
echo "GRAD_CLIP=$GRAD_CLIP"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * SLURM_NTASKS))"
echo "========================================="

mkdir -p results logs

# ============ TRAINING ============
echo "Starting training with srun..."
echo "========================================="

srun --wait=60 \
     --kill-on-bad-exit=1 \
     --cpu-bind=socket \
     python -u scripts/train.py \
        --region_len   $REGION_LEN \
        --nBins        $NBINS \
        --tracks       $TRACKS \
        --tracks_dir   tracks \
        --save_path    results \
        --biotype      $BIOTYPE \
        --model-type   $MODEL \
        --batch-size   $BATCH_SIZE \
        --max-epochs   $MAX_EPOCHS \
        --n_heads      $NHEADS \
        --dropout      $DROPOUT \
        --learning_rate $LR \
        --weight_decay  $WEIGHT_DECAY \
        --warmup_steps  $WARMUP \
        --grad_accum   $GRAD_ACCUM \
        --grad_clip    $GRAD_CLIP \
     2>&1

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "========================================="
    echo "Training completed successfully!"
    echo "========================================="

    # ============ TESTING ============
    # Find the best checkpoint from training
    SAVE_DIR=$(ls -td results/model_${REGION_LEN}_${NBINS}_* 2>/dev/null | head -n 1)
    BEST_CKPT=$(ls -t "${SAVE_DIR}"/epoch=*.ckpt 2>/dev/null | head -n 1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT="${SAVE_DIR}/last.ckpt"
    fi
    echo "Testing with checkpoint: $BEST_CKPT"
    echo "========================================="

    srun --wait=60 \
         --kill-on-bad-exit=1 \
         --cpu-bind=socket \
         python -u scripts/train.py \
            --region_len   $REGION_LEN \
            --nBins        $NBINS \
            --tracks       $TRACKS \
            --tracks_dir   tracks \
            --save_path    results \
            --biotype      $BIOTYPE \
            --model-type   $MODEL \
            --batch-size   $BATCH_SIZE \
            --max-epochs   $MAX_EPOCHS \
            --n_heads      $NHEADS \
            --dropout      $DROPOUT \
            --learning_rate $LR \
            --weight_decay  $WEIGHT_DECAY \
            --warmup_steps  $WARMUP \
            --grad_accum   $GRAD_ACCUM \
            --grad_clip    $GRAD_CLIP \
            --checkpoint   "$BEST_CKPT" \
            --test \
         2>&1

    if [ $? -eq 0 ]; then
        echo "========================================="
        echo "Testing completed successfully!"
        echo "========================================="
    else
        echo "========================================="
        echo "Testing failed"
        echo "========================================="
        exit 1
    fi

else
    echo "========================================="
    echo "Training failed with exit code $TRAIN_EXIT"
    echo "========================================="
    exit 1
fi

echo "Job finished at: $(date)"
echo "========================================="
