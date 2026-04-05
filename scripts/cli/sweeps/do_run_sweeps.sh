#!/bin/bash
#SBATCH --job-name=sweeps          # Specify job name
#SBATCH --partition=gpu            # Specify partition name
#SBATCH --nodes=1                  # Specify number of nodes
#SBATCH --cpus-per-task=24         # Specify number of CPUs (cores) per task
#SBATCH --mem-per-cpu=8G           # Specify amount of memory per CPU
#SBATCH --gres=shard:1             # Request one GPU
#SBATCH --time=48:00:00            # Time each job array element is allowed to run, here 48 hours
#SBATCH --output=outs/run_sweeps.o%j     # File name for standard output
#SBATCH --error=outs/run_sweeps.e%j      # File name for standard error output
#SBATCH --array=0-7                     # 8 models * 3 datasets = 24 jobs (0-23)

# Run sweeps for each model and data combination using SLURM job arrays
# Each task in the job array corresponds to a unique (model, dataset) pair

# --- CONFIGURATION ---
date
WANDB_AGENT_MAX_INITIAL_FAILURES=10
NUM_RUNS=200

# Model and Dataset Definitions
models=(SimpleCNN BPNet my_cnn RNN VCNN VCNNBpnet CNN BPNetReal)
datas=(core_mean flank_mean all_mean)
datas=(combined_mean)

model=${models[$SLURM_ARRAY_TASK_ID % ${#models[@]}]}
data=${datas[$SLURM_ARRAY_TASK_ID / ${#models[@]}]}

wandb agent --count ${NUM_RUNS} btg/P09_TFBS_Sweep_${data}/$(cat ./sweep_ids/${model}_${data}_sweep_id.txt)
date