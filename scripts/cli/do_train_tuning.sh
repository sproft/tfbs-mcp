#!/bin/bash
#SBATCH --job-name=tune      # Specify job name
#SBATCH --partition=gpu    # Specify partition name
#SBATCH --nodes=1              # Specify number of nodes
#SBATCH --cpus-per-task=32     # Specify number of CPUs (cores) per task
#SBATCH --mem=400G       # Specify amount of memory per CPU
#SBATCH --gres=shard:1            # Request one GPU
#SBATCH --time=48:00:00        # Time each job array element is allowed to run, here one hour
#SBATCH --output=outs/tune.o%j    # File name for standard output
#SBATCH --error=outs/tune.e%j     # File name for standard error output
#SBATCH --array=0-2 # 3 model types (0-2)

models=(RNN VCNNBpnet BPNetReal)
model=${models[$SLURM_ARRAY_TASK_ID]}

datas=(core_median_24_rc flank_median_24_rc all_mean_nO_24_rc)
date
for data in "${datas[@]}"
do
  for norm in "standardize"
  do
    echo $data
    python train_cli.py fit \
    --model $model \
    --model.input_length 24 \
    --data.data_path /sc-projects/sc-proj-cc17-P09_TFBS/data/tensors/${data} \
    --data.batch_size 64 \
    --data.scaling_method ${norm} \
    --trainer yamls/trainers/default.yaml \
    --trainer.logger.init_args.project P09_${data}_${norm}_${model} \
    --trainer.callbacks.init_args.dirpath /sc-projects/sc-proj-cc17-P09_TFBS/saved_models_tuned/${data}/${norm} \
    --trainer.callbacks.init_args.filename best_${model}
  done
done
date