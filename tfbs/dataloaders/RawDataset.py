import torch
import pandas as pd
from typing import Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os # Import the os module for path manipulation

class CsvDataset(Dataset):
    """
    A Dataset for handling tabular data from a pandas DataFrame.

    This dataset converts features and labels into PyTorch tensors, ready for training.

    Attributes:
        features (torch.Tensor): The input features for the model.
        labels (torch.Tensor): The corresponding labels.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], target_col: str):
        """
        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature_cols (List[str]): A list of column names to be used as features.
            target_col (str): The name of the target/label column.
        """
        # Ensure data types are appropriate for tensor conversion
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        # Assuming classification, labels are converted to long integers
        self.labels = torch.tensor(df[target_col].values, dtype=torch.long)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Retrieves a single sample (features and label) from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A tuple containing the features and the label for the given index.
        """
        return self.features[idx], self.labels[idx]

# ---

class CsvDataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule for generic CSV files, with unbalanced data handling.

    This module handles data loading, splitting, and preprocessing (scaling)
    for a typical machine learning workflow with tabular data. It now includes
    options for addressing class imbalance.
    """
    def __init__(
            self,
            data_path: str,
            target_col: str = 'disease_causing',
            cols_to_drop: Optional[List[str]] = None,
            batch_size: int = 64,
            num_workers: int = 4,
            unbalanced_data_handling: str = 'none', # 'none', 'sampler', 'class_weights'
            test_data_output_path: Optional[str] = None): # New parameter to save test data
        """
        Args:
            data_path (str): Path to the input CSV file.
            target_col (str): The name of the target variable column.
            cols_to_drop (Optional[List[str]]): A list of column names to drop from the dataset.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses for data loading.
            unbalanced_data_handling (str): Strategy for handling unbalanced data.
                                            'none': No special handling.
                                            'sampler': Uses WeightedRandomSampler for training.
                                            'class_weights': Calculates and exposes class weights.
            test_data_output_path (Optional[str]): Path where the preprocessed test DataFrame
                                                    will be saved as a CSV file. If None, it's not saved.
        """
        super().__init__()
        # save_hyperparameters() allows us to access init args via self.hparams
        self.save_hyperparameters()

        # Placeholders for data and scaler
        self.df = None
        self.train_df, self.val_df, self.test_df = None, None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.feature_cols = None
        self.scaler = None
        self._class_weights = None # To store calculated class weights

    def prepare_data(self):
        """
        Called on a single process. Use this for downloading or initial parsing.
        Here, we load the main CSV file into a pandas DataFrame.
        """
        self.df = pd.read_csv(self.hparams.data_path)

    def setup(self, stage: Optional[str] = None):
        """
        Called on every process in DDP. Handles splitting, scaling, and dataset creation.
        Also calculates class weights or sample weights for unbalanced data handling.
        Finally, saves the processed test set if a path is provided.

        Args:
            stage (Optional[str]): Can be 'fit', 'validate', 'test', or 'predict'.
                                   Used to perform setup logic only when needed.
        """
        # Ensure data is loaded if setup is called directly (e.g., for testing)
        if self.df is None:
            self.prepare_data()

        # 1. Drop specified columns if any
        if self.hparams.cols_to_drop:
            self.df.drop(columns=self.hparams.cols_to_drop, inplace=True, errors='ignore')

        # 2. Identify feature columns (all columns except the target)
        if not self.feature_cols:
             self.feature_cols = [col for col in self.df.columns if col != self.hparams.target_col]

        # 3. Split data into training, validation, and test sets
        # First split into train_val and test
        train_val_df, self.test_df = train_test_split(
            self.df, test_size=0.2, shuffle=True, random_state=42, stratify=self.df[self.hparams.target_col]
        )
        # Then split train_val into train and val
        self.train_df, self.val_df = train_test_split(
            train_val_df, test_size=0.25, shuffle=True, random_state=42, stratify=train_val_df[self.hparams.target_col]
        ) # 0.25 of 0.8 is 0.2, so 60/20/20 split

        # 4. Fit and apply feature scaling
        # StandardScaler standardizes features by removing the mean and scaling to unit variance
        self.scaler = StandardScaler()

        # Fit the scaler ONLY on the training data to prevent data leakage
        self.scaler.fit(self.train_df[self.feature_cols])

        # Apply the transformation to all data splits
        self.train_df[self.feature_cols] = self.scaler.transform(self.train_df[self.feature_cols])
        self.val_df[self.feature_cols] = self.scaler.transform(self.val_df[self.feature_cols])
        self.test_df[self.feature_cols] = self.scaler.transform(self.test_df[self.feature_cols])

        # 5. Instantiate the datasets for each split
        self.train_dataset = CsvDataset(self.train_df, self.feature_cols, self.hparams.target_col)
        self.val_dataset = CsvDataset(self.val_df, self.feature_cols, self.hparams.target_col)
        self.test_dataset = CsvDataset(self.test_df, self.feature_cols, self.hparams.target_col)

        # 6. Handle unbalanced data if specified
        if self.hparams.unbalanced_data_handling == 'class_weights':
            # Calculate class weights for the training set
            class_weights_array = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(self.train_df[self.hparams.target_col]),
                y=self.train_df[self.hparams.target_col]
            )
            self._class_weights = torch.tensor(class_weights_array, dtype=torch.float32)

        elif self.hparams.unbalanced_data_handling == 'sampler':
            # Calculate sample weights for WeightedRandomSampler
            class_counts = self.train_df[self.hparams.target_col].value_counts()
            # Invert class frequencies to give more weight to minority classes
            weights = 1. / class_counts
            self.sample_weights = self.train_df[self.hparams.target_col].apply(lambda x: weights[x]).values

        # 7. Save the holdout test dataset if a path is provided
        if self.hparams.test_data_output_path:
            # Ensure the directory exists
            output_dir = os.path.dirname(self.hparams.test_data_output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.test_df.to_csv(self.hparams.test_data_output_path, index=False)
            print(f"Test dataset saved to: {self.hparams.test_data_output_path}")


    @property
    def class_weights(self):
        """
        Returns the calculated class weights if 'class_weights' handling is enabled.
        Otherwise, returns None. These can be used in your model's loss function.
        """
        return self._class_weights

    def train_dataloader(self):
        """Returns the DataLoader for the training set, potentially with a WeightedRandomSampler."""
        if self.hparams.unbalanced_data_handling == 'sampler' and hasattr(self, 'sample_weights'):
            sampler = WeightedRandomSampler(
                num_samples=len(self.train_dataset),
                weights=self.sample_weights,
                replacement=True # Allows sampling the same sample multiple times
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                sampler=sampler, # Use sampler instead of shuffle=True
                num_workers=self.hparams.num_workers,
                persistent_workers=True,
                pin_memory=True
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=True, # Default shuffling if no sampler
                num_workers=self.hparams.num_workers,
                persistent_workers=True,
                pin_memory=True
            )

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False, # Typically no shuffling for validation
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False, # Typically no shuffling for testing
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=True
        )
