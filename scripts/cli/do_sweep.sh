#!/bin/bash
#SBATCH --job-name=sweep      # Specify job name
#SBATCH --partition=gpu    # Specify partition name
#SBATCH --nodes=1              # Specify number of nodes
#SBATCH --cpus-per-task=24     # Specify number of CPUs (cores) per task
#SBATCH --mem=400G       # Specify amount of memory per CPU
#SBATCH --gres=shard:1            # Request one GPU
#SBATCH --time=48:00:00        # Time each job array element is allowed to run, here one hour
#SBATCH --output=outs/sweep.o%j    # File name for standard output
#SBATCH --error=outs/sweep.e%j     # File name for standard error output

wandb agent btg/P09x_core_0.1/cg4nemyr