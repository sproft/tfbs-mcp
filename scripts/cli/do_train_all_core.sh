#!/bin/bash
#SBATCH --job-name=train_core      # Specify job name
#SBATCH --partition=gpu    # Specify partition name
#SBATCH --nodes=1              # Specify number of nodes
#SBATCH --cpus-per-task=24     # Specify number of CPUs (cores) per task
#SBATCH --mem-per-cpu=8G       # Specify amount of memory per CPU
#SBATCH --gres=shard:1            # Request one GPU
#SBATCH --time=48:00:00        # Time each job array element is allowed to run, here one hour
#SBATCH --output=outs/train_core.o%j    # File name for standard output
#SBATCH --error=outs/train_core.e%j     # File name for standard error output
#SBATCH --array=0-9

# Data, checkpoints and results live under $TFBS_PROJECT_ROOT. Default to the
# repo root (two levels up from this script) so a fresh clone runs unedited.
# Data, checkpoints and results live under $TFBS_PROJECT_ROOT.  Derive it if
# unset: the script's own location works for a direct run, but under sbatch
# Slurm executes a COPY in its spool dir, so fall back to the submit directory.
# Each candidate must actually look like the repo, otherwise we would silently
# train against a wrong (but existing) path such as /var/spool.
if [ -z "${TFBS_PROJECT_ROOT}" ]; then
  for _cand in "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"                "$(cd "${SLURM_SUBMIT_DIR:-$PWD}/../.." 2>/dev/null && pwd)"                "${SLURM_SUBMIT_DIR:-$PWD}"; do
    if [ -n "${_cand}" ] && [ -d "${_cand}/tfbs" ] && [ -d "${_cand}/scripts" ]; then
      TFBS_PROJECT_ROOT="${_cand}"
      break
    fi
  done
fi
if [ -z "${TFBS_PROJECT_ROOT}" ]; then
  echo "ERROR: could not locate the tfbs-mcp checkout. Set TFBS_PROJECT_ROOT." >&2
  exit 1
fi

models=(SimpleCNN MLP BPNet my_cnn RNN EquiNet VCNN VCNNBpnet CNN BPNetReal)
model=${models[$SLURM_ARRAY_TASK_ID]}

datas=(core_0.1_24 core_0.2_24 core_0.1_24_rc core_0.2_24_rc)
date
for data in "${datas[@]}"
do
  for norm in "normalize" "standardize" "none"
  do
    echo $data
    python train_cli.py fit \
    --model $model \
    --model.input_length 24 \
    --data.data_path ${TFBS_PROJECT_ROOT}/data/tensors/${data} \
    --data.batch_size 64 \
    --data.scaling_method ${norm} \
    --trainer yamls/trainers/default.yaml \
    --trainer.logger.init_args.project P09_${data}_${norm} \
    --trainer.callbacks.init_args.dirpath ${TFBS_PROJECT_ROOT}/saved_models/${data}/${norm} \
    --trainer.callbacks.init_args.filename best_${model}
  done
done
date
