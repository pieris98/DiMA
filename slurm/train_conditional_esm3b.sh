#!/bin/bash
#SBATCH --job-name=dima-conditional-esm3b
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/dima_conditional_esm3b_%j.out
#SBATCH --error=logs/dima_conditional_esm3b_%j.err

# ============================================================
# DiMA Conditional Training - ESM-3B on Leonardo HPC
# SwissProt Dataset
# ============================================================

# Ensure Pixi is in the path
export PATH="$SCRATCH/.pixi/bin:$PATH"

# Set WandB to offline mode
export WANDB_MODE=offline

# Distributed training variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=31345
export GPUS_PER_NODE=4
export NNODES=$SLURM_NNODES
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# Working directories
export PROJECT_ROOT=/home/cherry/dev/phd/DiMA
export WORK_DIR=$SCRATCH/dima_work
export DATA_DIR=$WORK_DIR/data
export CHECKPOINT_DIR=$WORK_DIR/checkpoints

# Create working directories
mkdir -p $WORK_DIR/logs
mkdir -p $DATA_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p $WORK_DIR/generated_sequences
mkdir -p $WORK_DIR/wandb

# Copy data to SCRATCH if not already there
if [ ! -d "$DATA_DIR/swissprot" ]; then
    echo "Copying SwissProt dataset to SCRATCH..."
    cp -r $PROJECT_ROOT/data/swissprot $DATA_DIR/
fi

if [ ! -d "$DATA_DIR/distributions" ]; then
    mkdir -p $DATA_DIR/distributions
    cp $PROJECT_ROOT/data/distributions/*.npy $DATA_DIR/distributions/
fi

# Copy checkpoint directory structure
if [ ! -d "$CHECKPOINT_DIR/statistics" ]; then
    cp -r $PROJECT_ROOT/checkpoints/statistics $CHECKPOINT_DIR/
fi

# Change to project directory
cd $PROJECT_ROOT

echo "=========================================="
echo "DiMA Conditional Training - ESM-3B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $NNODES ($SLURM_JOB_NODELIST)"
echo "Total GPUs: $WORLD_SIZE"
echo "Start time: $(date)"
echo "=========================================="

# Run training with torchrun for multi-node distributed training
torchrun \
    --nnodes=$NNODES \
    --node_rank=$SLURM_PROCID \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pixi run -e dima-env python train_diffusion.py \
        --config-name config \
        encoder=esm2 \
        datasets=swissprot \
        datasets.data_dir=$DATA_DIR/swissprot \
        datasets.length_distribution=$DATA_DIR/distributions/swissprot.npy \
        project.path=$WORK_DIR \
        project.checkpoints_folder=$CHECKPOINT_DIR \
        project.statistics_folder=$CHECKPOINT_DIR/statistics \
        project.wandb_project=dima-conditional-esm3b \
        ddp.enabled=true \
        training.training_iters=100000 \
        training.eval_interval=10000 \
        training.save_interval=10000 \
        training.batch_size=256

echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="
