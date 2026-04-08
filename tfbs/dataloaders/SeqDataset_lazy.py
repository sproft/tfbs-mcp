import torch
import pandas as pd
from typing import Optional
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

# All other helper functions from your original code (like dinucleotide_shuffle, etc.)
# can be kept here if you need them for other purposes.

class LazySeqDataset(Dataset):
    """
    A "lazy" Dataset for one-hot-encoded sequences.
    
    This class is highly memory-efficient because it stores sequences as raw strings
    and performs one-hot encoding for a single item only when it's requested by the
    DataLoader (in `__getitem__`).
    
    Attributes:
        seqs (np.ndarray): Array of DNA sequence strings.
        labels (torch.Tensor): Tensor of corresponding labels.
        max_len (int): The length to which all sequences will be padded.
        nuc_map (dict): A mapping from nucleotide characters to integer indices.
    """
    def __init__(self, df: pd.DataFrame, seq_col: str = 'seq', target_col: str = 'log2FC', max_len: Optional[int] = None):
        self.seqs = df[seq_col].values
        self.labels = torch.tensor(df[target_col].values, dtype=torch.float32)

        # Determine max length for padding if not provided
        self.max_len = max_len if max_len is not None else len(max(self.seqs, key=len))
        self.nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx: int):
        """
        Retrieves, pads, and one-hot encodes a single sequence.
        This method is called by the DataLoader for each item in a batch.
        """
        seq_str = self.seqs[idx]
        label = self.labels[idx].unsqueeze(0) # Add dimension for consistency

        # 1. Pad sequence string to the uniform max_len
        padded_seq = seq_str.ljust(self.max_len, 'N')
        
        # 2. Map nucleotide characters to their integer indices
        indices = torch.tensor([self.nuc_map.get(nuc, 4) for nuc in padded_seq], dtype=torch.long)
        
        # 3. One-hot encode using PyTorch's efficient functional
        ohe_seq = F.one_hot(indices, num_classes=5).float()
        
        # 4. Remove the 'N' dimension and permute to (channels, length) for CNNs
        # Final shape will be (4, max_len)
        ohe_seq = ohe_seq[..., :4].permute(1, 0)
        
        return ohe_seq, label

# ---

class TFBSDataModule(LightningDataModule):
    """
    An optimized PyTorch Lightning DataModule for TFBS data.
    
    This module uses a lazy-loading dataset to handle large files efficiently. It also
    provides built-in support for label standardization and normalization.
    """
    def __init__(
            self,
            data_path: str,
            batch_size: int = 64,
            num_workers: int = 4, # A more common default
            scaling_method: Optional[str] = None):
        """
        Args:
            data_path (str): Path to the input CSV file.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            scaling_method (Optional[str]): Label scaling method. Accepts 'standardize', 
                                            'normalize', or None.
        """
        super().__init__()
        self.save_hyperparameters()

        # Placeholders for dataframes, datasets, and scaling statistics
        self.df = None
        self.train_df, self.val_df, self.test_df = None, None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.mean, self.std, self.min, self.max = None, None, None, None

    def prepare_data(self):
        """
        Called only on a single process. Use this for downloading or initial parsing.
        Here, we load the main CSV file into a pandas DataFrame.
        """
        self.df = pd.read_csv(self.hparams.data_path)
        self.df.columns = ["seq", "log2FC"]

    def setup(self, stage: Optional[str] = None):
        """
        Called on every process in DDP. Handles splitting, scaling, and dataset creation.
        """
        # 1. Split dataframes
        train_val_df, self.test_df = train_test_split(self.df, test_size=0.2, shuffle=True, random_state=42)
        self.train_df, self.val_df = train_test_split(train_val_df, test_size=0.2, shuffle=True, random_state=42)
        
        # 2. Calculate and apply scaling if requested
        if self.hparams.scaling_method:
            # Calculate stats ONLY from the training set's labels to prevent data leakage
            labels = torch.tensor(self.train_df['log2FC'].values, dtype=torch.float32)
            
            if self.hparams.scaling_method == 'standardize':
                self.mean = torch.mean(labels)
                self.std = torch.std(labels)
                # Apply transformation to all splits
                self.train_df.loc[:, 'log2FC'] = (self.train_df['log2FC'] - self.mean) / self.std
                self.val_df.loc[:, 'log2FC'] = (self.val_df['log2FC'] - self.mean) / self.std
                self.test_df.loc[:, 'log2FC'] = (self.test_df['log2FC'] - self.mean) / self.std
            
            elif self.hparams.scaling_method == 'normalize':
                self.min = torch.min(labels)
                self.max = torch.max(labels)
                # Apply transformation to all splits
                self.train_df.loc[:, 'log2FC'] = (self.train_df['log2FC'] - self.min) / (self.max - self.min)
                self.val_df.loc[:, 'log2FC'] = (self.val_df['log2FC'] - self.min) / (self.max - self.min)
                self.test_df.loc[:, 'log2FC'] = (self.test_df['log2FC'] - self.min) / (self.max - self.min)
            
            else:
                raise ValueError(f"Unknown scaling_method: '{self.hparams.scaling_method}'")

        # 3. Determine max sequence length from the training set for consistent padding
        max_len = len(max(self.train_df['seq'], key=len))

        # 4. Instantiate the lazy datasets for each split
        self.train_dataset = LazySeqDataset(self.train_df, max_len=max_len)
        self.val_dataset = LazySeqDataset(self.val_df, max_len=max_len)
        self.test_dataset = LazySeqDataset(self.test_df, max_len=max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True)
    
    def reverse_transform(self, scaled_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverses the applied scaling on a tensor of labels or predictions. Useful for inference.
        """
        if self.hparams.scaling_method == 'standardize':
            if self.mean is None or self.std is None:
                raise RuntimeError("Scaling params not computed. Run setup() first.")
            return (scaled_tensor * self.std) + self.mean
        elif self.hparams.scaling_method == 'normalize':
            if self.min is None or self.max is None:
                raise RuntimeError("Scaling params not computed. Run setup() first.")
            return scaled_tensor * (self.max - self.min) + self.min
        else:
            return scaled_tensor