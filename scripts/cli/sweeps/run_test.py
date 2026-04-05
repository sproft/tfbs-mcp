# run.py  
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], required=True)
args = parser.parse_args()

# Initialize a W&B Run
with wandb.init('test-project') as run:
    run.log({'validation_loss':1})
