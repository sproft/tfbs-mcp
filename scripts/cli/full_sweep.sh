#!/bin/bash
#SBATCH --job-name=sweep     # Specify job name
#SBATCH --partition=gpu      # Specify partition name
#SBATCH --nodes=1            # Specify number of nodes
#SBATCH --cpus-per-task=24   # Specify number of CPUs (cores) per task
#SBATCH --mem-per-cpu=8G     # Specify amount of memory per CPU
#SBATCH --gres=shard:1       # Request one GPU
#SBATCH --time=48:00:00      # Time each job array element is allowed to run, here one hour
#SBATCH --output=outs/sweep.o%j     # File name for standard output
#SBATCH --error=outs/sweep.e%j      # File name for standard error output
#SBATCH --array=0-8

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

# --- CONFIGURATION ---

# Model and Dataset Definitions
models=(RNN VCNNBpnet BPNetReal)
data_names=(core_median_24_rc flank_median_24_rc all_mean_nO_24_rc)
scaling_method="standardize" # Fixed for the initial sweep as per your earlier config

# Base Configuration
WANDB_SWEEP_BASE_DIR="./wandb_sweep_ids"
WANDB_SWEEP_ID_SUFFIX="_sweep_id.txt"
WANDB_SWEEP_ID_FULL=""
SWEEP_CONFIG="wandb_sweep_config.yaml"

# Define the base checkpoint directory
CHECKPOINT_DIR="${TFBS_PROJECT_ROOT}/saved_models_tuned"

# --- CALCULATE TASK PARAMETERS ---

# Total number of models and datasets
NUM_MODELS=${#models[@]}
NUM_DATASETS=${#data_names[@]}

# Determine the index for the model and dataset based on the SLURM_ARRAY_TASK_ID
MODEL_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_MODELS ))
DATASET_INDEX=$(( SLURM_ARRAY_TASK_ID / NUM_MODELS ))

# Assign the specific model and dataset for this job
MODEL=${models[$MODEL_INDEX]}
DATA_NAME=${data_names[$DATASET_INDEX]}

# Construct the full data path and project name
DATA_PATH="${TFBS_PROJECT_ROOT}/data/tensors/${DATA_NAME}"
WANDB_PROJECT="P09_TFBS_Sweep_${DATA_NAME}"

# Define the dataset-specific file path for the sweep ID
SWEEP_ID_FILE="${WANDB_SWEEP_BASE_DIR}/${DATA_NAME}${WANDB_SWEEP_ID_SUFFIX}"

# Ensure the directory for sweep ID files exists
mkdir -p "$WANDB_SWEEP_BASE_DIR"


# --- SWEEP INITIALIZATION LOGIC (Per Dataset) ---

# 1. INITIALIZE SWEEP (Only run this on the first task for each dataset group, i.e., MODEL_INDEX=0. Tasks 0, 3, 6)
if [ $MODEL_INDEX -eq 0 ]; then
    if [ ! -f "$SWEEP_ID_FILE" ]; then
        echo "Task $SLURM_ARRAY_TASK_ID (Initializer for $DATA_NAME): Creating new WandB sweep in project $WANDB_PROJECT from $SWEEP_CONFIG..."
        
        # Command to create the sweep and capture the full ID (entity/project/id)
        # NOTE: The project name is now dynamic ($WANDB_PROJECT)
        WANDB_INIT_OUTPUT=$(wandb sweep --project $WANDB_PROJECT $SWEEP_CONFIG 2>&1)
        
        # Extract the full sweep path (entity/project/id) from the 'wandb agent' command line
        # This is more robust than parsing the 'Created sweep' line.
        WANDB_SWEEP_ID_FULL=$(echo "$WANDB_INIT_OUTPUT" | grep 'Creating sweep with ID' | awk '{print $NF}')
        
        # CRITICAL FIX: Clean the extracted ID from any invisible whitespace/newlines
        WANDB_SWEEP_ID_FULL=$(echo "$WANDB_SWEEP_ID_FULL" | tr -d '[:space:]')
        
        if [ -z "$WANDB_SWEEP_ID_FULL" ]; then
            echo "ERROR: Failed to create WandB sweep for dataset $DATA_NAME. Check wandb login and $SWEEP_CONFIG."
            echo "$WANDB_INIT_OUTPUT"
            exit 1
        fi

        # Save the ID using the dataset-specific file name for subsequent tasks of this group
        # The ID is already cleaned, but using echo > file usually adds a newline, 
        # which is why we clean it again on load (step 3).
        echo "$WANDB_SWEEP_ID_FULL" > "$SWEEP_ID_FILE"
        echo "Created and saved WandB Sweep ID: $WANDB_SWEEP_ID_FULL for $DATA_NAME"
    else
        echo "Task $SLURM_ARRAY_TASK_ID (Initializer for $DATA_NAME): Sweep ID file ($SWEEP_ID_FILE) already exists. Skipping creation."
    fi
fi

# 2. Wait for array task 0, 3, or 6 to finish sweep creation and file write for its respective dataset
# This sleep is crucial to prevent a race condition where later tasks try to read the file before the initializer has written it.
if [ $MODEL_INDEX -eq 0 ]; then
    # Initializer tasks don't need to wait for themselves, but a short pause ensures the file is flushed
    echo "Initializer task $SLURM_ARRAY_TASK_ID finished creation."
else
    # Non-initializer tasks must wait for the initializer task of their group (e.g., 1 waits for 0, 4 waits for 3)
    echo "Task $SLURM_ARRAY_TASK_ID waiting 15 seconds for sweep creation file..."
    sleep 15
fi

# 3. All tasks (0-8) read the ID from the dataset-specific file
if [ -f "$SWEEP_ID_FILE" ]; then
    WANDB_SWEEP_ID=$(cat "$SWEEP_ID_FILE")
    # CRITICAL FIX: Clean the loaded ID from any invisible whitespace/newlines 
    # that might have been saved by the 'echo >' command.
    WANDB_SWEEP_ID=$(echo "$WANDB_SWEEP_ID" | tr -d '[:space:]') 
    
    echo "Loaded WandB Sweep ID: $WANDB_SWEEP_ID for $DATA_NAME"
else
    echo "FATAL ERROR: WandB Sweep ID file ($SWEEP_ID_FILE) not found. Initializer task for $DATA_NAME might have failed or not finished."
    exit 1
fi

# --- EXECUTION ---

date
echo "Starting job array task $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL | Dataset: $DATA_NAME | Scaling: $scaling_method"
echo "Data Path: $DATA_PATH"
echo "WandB Project: $WANDB_PROJECT"

# Call run_sweep.py. The Python script will connect to the specified SWEEP_ID
# and run the WandB agent for 50 trials, using the fixed parameters passed here.
# Note: The quotes around variables are essential to prevent shell splitting 
# when spaces are involved, though the cleaned ID should be space-free now.
python run_sweep.py \
    --model "$MODEL" \
    --data-path "$DATA_PATH" \
    --data-name "$DATA_NAME" \
    --scaling-method "$scaling_method" \
    --sweep-id "$WANDB_SWEEP_ID" \
    --n-trials 50 \
    --batch-size 64 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --project-name "$WANDB_PROJECT"

date
