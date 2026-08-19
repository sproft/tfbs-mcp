#!/bin/bash
#SBATCH --job-name=final
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=outs/final.o%A_%a
#SBATCH --error=outs/final.e%A_%a
#SBATCH --array=0-5%6    # run tasks 0..5, limit concurrent tasks to 6

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

date

# list of commands (one per array index)
commands=(
'python ${TFBS_PROJECT_ROOT}/scripts/cli/train_cli.py fit --model=VCNNBpnet --model.input_length=24 --data.data_path=${TFBS_PROJECT_ROOT}/data/tensors/all_mean --data.scaling_method=standardize --trainer=${TFBS_PROJECT_ROOT}/scripts/cli/yamls/trainers/default.yaml --trainer.logger.init_args.project=P09_all_mean_standardize_VCNNBpnet_final --trainer.callbacks.init_args.dirpath=${TFBS_PROJECT_ROOT}/saved_models_final/all_mean/standardize --trainer.callbacks.init_args.filename=best_VCNNBpnet "--model.dense_sizes=[128, 64]" "--model.dilations=[1, 1, 2, 4, 8]" --model.kernel_size=42 --model.num_channels=82 --model.pool_output_size=167'
'python ${TFBS_PROJECT_ROOT}/scripts/cli/train_cli.py fit --model=RNN --model.input_length=24 --data.data_path=${TFBS_PROJECT_ROOT}/data/tensors/all_mean --data.scaling_method=standardize --trainer=${TFBS_PROJECT_ROOT}/scripts/cli/yamls/trainers/default.yaml --trainer.logger.init_args.project=P09_all_mean_standardize_RNN_final --trainer.callbacks.init_args.dirpath=${TFBS_PROJECT_ROOT}/saved_models_final/all_mean/standardize --trainer.callbacks.init_args.filename=best_RNN --model.bidirectional=True --model.conv_out_channels=115 --model.dense_size=193 --model.dropout_conv=0.30385810801659574 --model.dropout_fc=0.42671878718015466 --model.gru_hidden_size=57 --model.gru_num_layers=2 --model.kernel_size=7 --model.pool_size=3'
'python ${TFBS_PROJECT_ROOT}/scripts/cli/train_cli.py fit --model=VCNNBpnet --model.input_length=24 --data.data_path=${TFBS_PROJECT_ROOT}/data/tensors/flank_mean --data.scaling_method=standardize --trainer=${TFBS_PROJECT_ROOT}/scripts/cli/yamls/trainers/default.yaml --trainer.logger.init_args.project=P09_flank_mean_standardize_VCNNBpnet_final --trainer.callbacks.init_args.dirpath=${TFBS_PROJECT_ROOT}/saved_models_final/flank_mean/standardize --trainer.callbacks.init_args.filename=best_VCNNBpnet "--model.dense_sizes=[128, 64]" "--model.dilations=[1, 1, 2, 4, 8]" --model.kernel_size=13 --model.num_channels=120 --model.pool_output_size=151'
'python ${TFBS_PROJECT_ROOT}/scripts/cli/train_cli.py fit --model=RNN --model.input_length=24 --data.data_path=${TFBS_PROJECT_ROOT}/data/tensors/flank_mean --data.scaling_method=standardize --trainer=${TFBS_PROJECT_ROOT}/scripts/cli/yamls/trainers/default.yaml --trainer.logger.init_args.project=P09_flank_mean_standardize_RNN_final --trainer.callbacks.init_args.dirpath=${TFBS_PROJECT_ROOT}/saved_models_final/flank_mean/standardize --trainer.callbacks.init_args.filename=best_RNN --model.bidirectional=False --model.conv_out_channels=173 --model.dense_size=193 --model.dropout_conv=0.0787679531434694 --model.dropout_fc=0.12782906842365138 --model.gru_hidden_size=65 --model.gru_num_layers=2 --model.kernel_size=6 --model.pool_size=4'
'python ${TFBS_PROJECT_ROOT}/scripts/cli/train_cli.py fit --model=VCNNBpnet --model.input_length=24 --data.data_path=${TFBS_PROJECT_ROOT}/data/tensors/core_mean --data.scaling_method=standardize --trainer=${TFBS_PROJECT_ROOT}/scripts/cli/yamls/trainers/default.yaml --trainer.logger.init_args.project=P09_core_mean_standardize_VCNNBpnet_final --trainer.callbacks.init_args.dirpath=${TFBS_PROJECT_ROOT}/saved_models_final/core_mean/standardize --trainer.callbacks.init_args.filename=best_VCNNBpnet "--model.dense_sizes=[128, 64]" "--model.dilations=[1, 1, 2, 4, 8]" --model.kernel_size=13 --model.num_channels=82 --model.pool_output_size=178'
'python ${TFBS_PROJECT_ROOT}/scripts/cli/train_cli.py fit --model=RNN --model.input_length=24 --data.data_path=${TFBS_PROJECT_ROOT}/data/tensors/core_mean --data.scaling_method=standardize --trainer=${TFBS_PROJECT_ROOT}/scripts/cli/yamls/trainers/default.yaml --trainer.logger.init_args.project=P09_core_mean_standardize_RNN_final --trainer.callbacks.init_args.dirpath=${TFBS_PROJECT_ROOT}/saved_models_final/core_mean/standardize --trainer.callbacks.init_args.filename=best_RNN --model.bidirectional=True --model.conv_out_channels=169 --model.dense_size=161 --model.dropout_conv=0.34922360143123676 --model.dropout_fc=0.20123656565325365 --model.gru_hidden_size=110 --model.gru_num_layers=1 --model.kernel_size=10 --model.pool_size=3'
)

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=0-5 to parallelize, or run the script as a single job."
  exit 1
fi

task_id=$SLURM_ARRAY_TASK_ID
cmd="${commands[$task_id]}"

echo "Starting task $task_id on host $(hostname) (SLURM_JOB_ID=$SLURM_JOB_ID)"
echo "Command: $cmd"
eval "$cmd"

date