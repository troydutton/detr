#!/bin/bash
#SBATCH -J detr
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -p gh
#SBATCH -t 48:00:00
#SBATCH -o /work/09889/troydutton/vista/out/detr/detr-%j
#SBATCH -A OTH25007

# Worker Logic
if [[ "$1" == "--worker" ]]; then
   	# Environment setup 
	source ~/.bashrc
	module load gcc/14.2.0 cuda/12.4 nvidia_math/12.4 cmake/3.29.5
	conda activate detr > /dev/null 2>&1
	cd ~/work/programs/detr/

	# Distributed variables
	RANK=$SLURM_PROCID
	WORLD_SIZE=$SLURM_NNODES
	export $OMP_NUM_THREADS=64

	echo "$((RANK + 1))/$WORLD_SIZE connecting to $MASTER_ADDR:$MASTER_PORT."

	accelerate launch \
		--multi_gpu \
		--num_machines=$WORLD_SIZE \
		--num_processes=$WORLD_SIZE \
		--machine_rank=$RANK \
		--main_process_ip=$MASTER_ADDR \
		--main_process_port=$MASTER_PORT \
		--mixed_precision bf16 \
		--rdzv_backend static \
		src/train.py

	exit 0
fi

# Launcher Logic
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=1234

srun ~/work/programs/detr/scripts/train-vista.slurm --worker
