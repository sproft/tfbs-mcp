#!/bin/bash
#SBATCH --job-name=train_all      # Specify job name
#SBATCH --partition=gpu    # Specify partition name
#SBATCH --nodes=1              # Specify number of nodes
#SBATCH --cpus-per-task=32     # Specify number of CPUs (cores) per task
#SBATCH --mem=400G       # Specify amount of memory per CPU
#SBATCH --gres=shard:2            # Request one GPU
#SBATCH --time=48:00:00        # Time each job array element is allowed to run, here one hour
#SBATCH --output=outs/train_all.o%j    # File name for standard output
#SBATCH --error=outs/train_all.e%j     # File name for standard error output
#SBATCH --array=0

models=(EquiNet)
model=${models[$SLURM_ARRAY_TASK_ID]}

datas=(all_0.5_subtraction)
date
for data in "${datas[@]}"
do
  echo $data
  python train_cli.py fit \
    --model $model \
    --model.input_length 70 \
    --data.data_path /sc-projects/sc-proj-cc17-P09_TFBS/data/tensors/${data} \
    --data.batch_size 8 \
    --trainer yamls/trainers/default.yaml \
    --trainer.logger.init_args.project P09_${data} \
    --trainer.callbacks.init_args.dirpath /sc-projects/sc-proj-cc17-P09_TFBS/saved_models/${data} \
    --trainer.callbacks.init_args.filename best_${model}_{epoch:02d}-{val_loss:.4f}
  date
done