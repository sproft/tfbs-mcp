# Create a sweep for each file in scripts/cli/sweeps/generated/
# Each file is named sweep_<model>_<data>.yaml
# The sweep ID is saved in scripts/cli/sweeps/sweep_ids/<model>_<data>_sweep_id.txt


for model in SimpleCNN BPNet my_cnn RNN VCNN VCNNBpnet CNN BPNetReal; do
  for data in core_mean flank_mean all_mean; do
    sweep_file="generated/sweep_${model}_${data}.yaml"
    sweep_id_file="sweep_ids/${model}_${data}_sweep_id.txt"
    
    if [ -f "$sweep_file" ]; then
      echo "Creating sweep for model $model and data $data using config $sweep_file"
      WANDB_INIT_OUTPUT=$(wandb sweep --project P09_TFBS_Sweep_$data $sweep_file 2>&1)
        
      WANDB_SWEEP_ID_FULL=$(echo "$WANDB_INIT_OUTPUT" | grep 'Creating sweep with ID' | awk '{print $NF}')
        
      # CRITICAL FIX: Clean the extracted ID from any invisible whitespace/newlines
      WANDB_SWEEP_ID_FULL=$(echo "$WANDB_SWEEP_ID_FULL" | tr -d '[:space:]')

      echo "$WANDB_SWEEP_ID_FULL" > "$sweep_id_file"
      echo "Sweep ID saved to $sweep_id_file"
    else
      echo "Sweep file $sweep_file does not exist. Skipping."
    fi
  done
done