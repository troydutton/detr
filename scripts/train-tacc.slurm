#!/bin/bash
#SBATCH -J job-name
#SBATCH -N num-names
#SBATCH --ntasks-per-node=1
#SBATCH -p partition
#SBATCH -t wall-clock-time
#SBATCH -o output-path


# Worker Logic
if [[ "$1" == "--worker" ]]; then
   	# Environment setup 
	module restore detr
	source $(conda info --base)/etc/profile.d/conda.sh
	conda activate detr > /dev/null 2>&1
	cd /work/09889/troydutton/ls6/programs/detr

	# Distributed variables
	NUM_MACHINES=$SLURM_NNODES
	NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))
	MACHINE_RANK=$SLURM_NODEID

	echo "$((MACHINE_RANK + 1))/$NUM_MACHINES connecting to $MASTER_ADDR:$MASTER_PORT."

	accelerate launch \
		--multi_gpu \
		--num_machines=$NUM_MACHINES \
		--num_processes=$NUM_PROCESSES \
		--machine_rank=$MACHINE_RANK \
		--main_process_ip=$MASTER_ADDR \
		--main_process_port=$MASTER_PORT \
		--mixed_precision bf16 \
		--rdzv_backend static \
		src/train.py \
		--config-name=$SLURM_JOB_NAME \
	exit 0
fi

# Launcher Logic
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=1234
export GPUS_PER_NODE=3
export OMP_NUM_THREADS=32

srun /work/09889/troydutton/ls6/programs/detr/scripts/train-tacc.slurm --worker
