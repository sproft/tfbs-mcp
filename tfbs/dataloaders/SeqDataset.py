import os
from typing import Optional
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule
import torch.nn.functional as F
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

class TFBSDataModule(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int = 64,
            num_workers: int = 24,
            preprocess: bool = False,
            scaling_method: Optional[str] = None):
        """
        PyTorch Lightning DataModule for TFBS data.

        Args:
            data_path (str): Path to the data. If preprocess=True, path to CSV. 
                             If preprocess=False, path to directory with pre-saved tensors.
            batch_size (int): The batch size for the dataloaders.
            num_workers (int): The number of workers for the dataloaders.
            preprocess (bool): If True, preprocesses data from a CSV file. 
                               If False, loads pre-saved tensors.
            scaling_method (Optional[str]): The scaling method to apply to labels. 
                                            Options: 'standardize', 'normalize', or None.
        """
        super().__init__()
        # This saves all hyperparameters (init args) to self.hparams
        # for easy access and logging
        self.save_hyperparameters()

        # Placeholders for scaling statistics and datasets
        self.mean, self.std = None, None
        self.min, self.max = None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, stage: Optional[str] = None):
        """
        This method is called on every GPU. It handles data loading, splitting,
        and all transformations, including the new scaling logic.
        """
        # Load data based on whether preprocessing is needed
        if self.hparams.preprocess:
            # This block runs if we start from a raw CSV file
            df = pd.read_csv(self.hparams.data_path)
            df.columns = ["seq", "log2FC"]
            
            # Split data
            train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
            train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=True, random_state=42)
            
            # One-hot encode the datasets
            train_seqs, train_labels = self._one_hot_encode(train_df, seq_col='seq', target_col='log2FC')
            val_seqs, val_labels = self._one_hot_encode(val_df, seq_col='seq', target_col='log2FC')
            test_seqs, test_labels = self._one_hot_encode(test_df, seq_col='seq', target_col='log2FC')
            
        else:
            # This block runs if we load pre-saved tensors from a directory
            data_dir = self.hparams.data_path
            train_seqs = torch.load(f'{data_dir}/train/seqs.pt')
            train_labels = torch.load(f'{data_dir}/train/labels.pt')
            val_seqs = torch.load(f'{data_dir}/val/seqs.pt')
            val_labels = torch.load(f'{data_dir}/val/labels.pt')
            test_seqs = torch.load(f'{data_dir}/test/seqs.pt')
            test_labels = torch.load(f'{data_dir}/test/labels.pt')

        # --- SCALING LOGIC ---
        if self.hparams.scaling_method:
            print(f"Applying '{self.hparams.scaling_method}' scaling to labels.")
            
            if self.hparams.scaling_method == 'standardize':
                # Standardization (Z-score): (x - mean) / std
                self.mean = torch.mean(train_labels)
                self.std = torch.std(train_labels)
                
                train_labels = (train_labels - self.mean) / self.std
                val_labels = (val_labels - self.mean) / self.std
                test_labels = (test_labels - self.mean) / self.std
                
            elif self.hparams.scaling_method == 'normalize':
                # Normalization (Min-Max): (x - min) / (max - min)
                self.min = torch.min(train_labels)
                self.max = torch.max(train_labels)
                
                train_labels = (train_labels - self.min) / (self.max - self.min)
                val_labels = (val_labels - self.min) / (self.max - self.min)
                test_labels = (test_labels - self.min) / (self.max - self.min)

            elif self.hparams.scaling_method == 'none':
                # No scaling applied
                pass

            else:
                raise ValueError(f"Unknown scaling_method: '{self.hparams.scaling_method}'. "
                                 f"Choose from 'standardize', 'normalize', or None.")

            # Persist scaling parameters so they can be used at inference time
            # without re-loading the training data (e.g. by the MCP server).
            self._save_scaling_params()

        # Create TensorDatasets for the dataloaders
        self.train_dataset = TensorDataset(train_seqs, train_labels)
        self.val_dataset = TensorDataset(val_seqs, val_labels)
        self.test_dataset = TensorDataset(test_seqs, test_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True)
    
    def reverse_transform(self, scaled_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverses the applied scaling on a tensor of labels or predictions.
        
        Args:
            scaled_tensor (torch.Tensor): A tensor with scaled values.
            
        Returns:
            torch.Tensor: A tensor with values in the original scale.
        """
        if self.hparams.scaling_method == 'standardize':
            if self.mean is None or self.std is None:
                raise RuntimeError("Standardization params not computed. Run setup() first.")
            return (scaled_tensor * self.std) + self.mean
        elif self.hparams.scaling_method == 'normalize':
            if self.min is None or self.max is None:
                raise RuntimeError("Normalization params not computed. Run setup() first.")
            return scaled_tensor * (self.max - self.min) + self.min
        else:
            # If no scaling was applied, return the original tensor
            return scaled_tensor

    def _save_scaling_params(self):
        """Save scaling parameters to a JSON file next to the training data.

        This allows the MCP server (and other inference code) to reverse
        the label transform without re-loading the training set.
        """
        import json as _json

        data_dir = self.hparams.data_path
        if data_dir is None:
            return

        params: dict = {"scaling_method": self.hparams.scaling_method}
        if self.hparams.scaling_method == "standardize" and self.mean is not None:
            params["mean"] = float(self.mean)
            params["std"] = float(self.std)
        elif self.hparams.scaling_method == "normalize" and self.min is not None:
            params["min"] = float(self.min)
            params["max"] = float(self.max)

        path = os.path.join(str(data_dir), "scaling_params.json")
        try:
            with open(path, "w") as f:
                _json.dump(params, f, indent=2)
            print(f"Scaling parameters saved to {path}")
        except OSError as e:
            print(f"Warning: could not save scaling params: {e}")

    def _one_hot_encode(self, df, seq_col, target_col):
        """Helper function to one-hot encode sequences from a dataframe."""
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        seqs = list(df[seq_col].values)
        
        # Pad sequences to the same length to create a single tensor
        max_len = len(max(seqs, key=len))
        
        indices_list = []
        for seq in seqs:
            padded_seq = seq.ljust(max_len, 'N')
            indices_list.append([nuc_map.get(nuc, 4) for nuc in padded_seq])

        indices = torch.tensor(indices_list, dtype=torch.long)
        ohe_seqs = F.one_hot(indices, num_classes=5).float()
        
        # Remove the 'N' dimension and permute to (batch, channels, length)
        ohe_seqs = ohe_seqs[..., :4].permute(0, 2, 1)

        labels = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)
        return ohe_seqs, labels