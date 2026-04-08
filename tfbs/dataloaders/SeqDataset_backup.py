from typing import Optional
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule
import torch.nn.functional as F
import torch
import numpy as np
from Bio import SeqIO
import pandas as pd
import random
from sklearn.model_selection import train_test_split

#needs cleanup turn into a class just for dinucleotide encoding

class TFBSDataModule(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int = 64,
            num_workers: int = 24,
            preprocess: bool = False):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocess = preprocess

    def prepare_data(self):
        if self.preprocess:
            df = pd.read_csv(self.data_path)
            df.columns = ["seq", "log2FC"]
            train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
            train, val = train_test_split(train, test_size=0.2, shuffle=True, random_state=42)
            self.train_seqs, self.train_labels = one_hot_encode_dataset(train, seq_col='seq', target_col='log2FC')
            self.val_seqs, self.val_labels = one_hot_encode_dataset(val, seq_col='seq', target_col='log2FC')
            self.test_seqs, self.test_labels = one_hot_encode_dataset(test, seq_col='seq', target_col='log2FC')
        else:
            # Load the tensor from the data_path
            self.train_seqs = torch.load(self.data_path+'/train/seqs.pt', weights_only=True)
            self.train_labels = torch.load(self.data_path+'/train/labels.pt', weights_only=True)
            self.val_seqs = torch.load(self.data_path+'/val/seqs.pt', weights_only=True)
            self.val_labels = torch.load(self.data_path+'/val/labels.pt', weights_only=True)
            self.test_seqs = torch.load(self.data_path+'/test/seqs.pt', weights_only=True)
            self.test_labels = torch.load(self.data_path+'/test/labels.pt', weights_only=True)

    def train_dataloader(self):
        train_dataset = TensorDataset(self.train_seqs, self.train_labels)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        val_dataset = TensorDataset(self.val_seqs, self.val_labels)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        test_dataset = TensorDataset(self.test_seqs, self.test_labels)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

class SeqDatasetOHE(Dataset):
    '''
    Dataset for one-hot-encoded sequences
    '''
    def __init__(self,
                 df,
                 seq_col='seq',
                 target_col='count'
                ):

        self.seqs = df[seq_col].values
        self.labels = df[target_col].values

        # One-hot encode sequences and convert to a torch tensor
        nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        indices = [torch.tensor([nuc_to_idx.get(nuc, 4) for nuc in seq],dtype=torch.long) for seq in self.seqs]
        # Pad the sequences to the same length
        indices = torch.nn.utils.rnn.pad_sequence(indices, batch_first=True, padding_value=4)
        # 0: A, 1: C, 2: G, 3: T, 4: N
        ohe_seqs = F.one_hot(indices, num_classes=5).float()

        # Remove the base N by setting it to [0,0,0,0]
        ohe_seqs = ohe_seqs[:, :, :4]

        #swith last two dimensions
        self.ohe_seqs = ohe_seqs.permute(0,2,1)

        # pack the sequences into a tensor
        #self.ohe_seqs=torch.nn.utils.rnn.pack_padded_sequence(ohe_seqs, lengths=[len(seq) for seq in self.seqs], batch_first=True, enforce_sorted=False)

        # Convert labels to a torch tensor and add an extra dimension
        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.seqs)
    
    def __getitem__(self,idx):
        # Given an index, return a tuple of an X with it's associated Y
        # This is called inside DataLoader
        seq = self.ohe_seqs[idx]
        label = self.labels[idx]
        
        return seq, label
    


## Here is a custom defined Dataset object specialized for one-hot encoded DNA:
class SeqDatasetOHE_slow(Dataset):
    '''
    Dataset for one-hot-encoded sequences
    '''
    def __init__(self,
                 df,
                 seq_col='seq',
                 target_col='count'
                ):
        # +--------------------+
        # | Get the X examples |
        # +--------------------+
        # extract the DNA from the appropriate column in the df
        self.seqs = list(df[seq_col].values)
        self.seqs = list(df.seq.values)
        
        # one-hot encode sequences, then stack in a torch tensor
        self.ohe_seqs = torch.stack([torch.tensor(one_hot_encode(x),dtype=torch.float32) for x in self.seqs])
    
        # +------------------+
        # | Get the Y labels |
        # +------------------+
        self.labels = torch.tensor(list(df[target_col].values),dtype=torch.float32).unsqueeze(1)
        
    def __len__(self): return len(self.seqs)
    
    def __getitem__(self,idx):
        # Given an index, return a tuple of an X with it's associated Y
        # This is called inside DataLoader
        seq = self.ohe_seqs[idx]
        label = self.labels[idx]
        
        return seq, label
    
## Here is a custom defined Dataset object specialized for dincleotide counted sequences:

class SeqDatasetDNE(Dataset):
    def __init__(self,
                 df,
                 seq_col='seq',
                 target_col='count'
                ):
        # +--------------------+
        # | Get the X examples |
        # +--------------------+
        # extract the DNA from the appropriate column in the df
        self.seqs = list(df[seq_col].values)
        self.seq_len = len(self.seqs[0])
        
        # one-hot encode sequences, then stack in a torch tensor
        self.dne_seqs = torch.stack([torch.tensor(list(dinucleotide_encode(x).values()),dtype=torch.float32) for x in self.seqs])
    
        # +------------------+
        # | Get the Y labels |
        # +------------------+
        self.labels = torch.tensor(list(df[target_col].values),dtype=torch.float32).unsqueeze(1)
        
    def __len__(self): return len(self.seqs)
    
    def __getitem__(self,idx):
        # Given an index, return a tuple of an X with it's associated Y
        # This is called inside DataLoader
        seq = self.dne_seqs[idx]
        label = self.labels[idx]
        
        return seq, label
    
def prepare_fasta_dfs(fasta_file):
    """
    Given a FASTA file, return a DataFrame with the sequences and labels.
    The first half of the sequences are class 1 and the second half are class 0.
    """

    # Load sequences from the FASTA file
    seqs = [str(record.seq).upper() for record in SeqIO.parse(fasta_file, 'fasta')]

    seqs_shuffled = [dinucleotide_shuffle(x) for x in seqs]
    
    #create shuffled seqs
    seqs = seqs + seqs_shuffled

    # Create labels for the sequences the first half is class 1
    # and the second half is class 0
    labels = [1 for x in range(len(seqs)//2)] + [0 for x in range(len(seqs)//2)]

    # Create a DataFrame with the sequences and labels
    df = pd.DataFrame({'seq':seqs,'label':labels})
    return df


def dinucleotide_shuffle(sequence):
    """
    Perform dinucleotide shuffling on a DNA sequence.
    """
    if len(sequence) < 2:
        return sequence

    # Create a list of dinucleotides
    dinucleotides = [sequence[i:i+2] for i in range(0, len(sequence) - 1, 2)]
    
    # If the sequence length is odd, add the last nucleotide as a single element
    if len(sequence) % 2 != 0:
        dinucleotides.append(sequence[-1])
    
    # Shuffle the dinucleotides
    random.shuffle(dinucleotides)
    
    # Join the shuffled dinucleotides to form the shuffled sequence
    shuffled_sequence = ''.join(dinucleotides)
    
    return shuffled_sequence


# Encode seqeunces with one-hot encoding
def one_hot_encode(seq):
    """
    Given a DNA sequence, return its one-hot encoding
    """
    # Make sure seq has only allowed bases
    allowed = set("ACTGN")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(f"Sequence contains chars not in allowed DNA alphabet (ACGTN): {invalid}")
        
    # Dictionary returning one-hot encoding for each nucleotide 
    nuc_d = {'A':[1.0,0.0,0.0,0.0],
            'C':[0.0,1.0,0.0,0.0],
            'G':[0.0,0.0,1.0,0.0],
            'T':[0.0,0.0,0.0,1.0],
            'N':[0.0,0.0,0.0,0.0]}
    
    # Create a tensor from the nucleotide sequence
    vec = torch.tensor([nuc_d[nuc] for nuc in seq], dtype=torch.float32)
        
    return vec.T

def dinucleotide_encode(seq):
    """
    Given a DNA sequence, count the number of all possible dinucleotides.
    """
    # Initialize a dictionary to store dinucleotide counts
    dinuc_d = {
        'AA': 0, 'AC': 0, 'AG': 0, 'AT': 0, 'AN': 0,
        'CA': 0, 'CC': 0, 'CG': 0, 'CT': 0, 'CN': 0,
        'GA': 0, 'GC': 0, 'GG': 0, 'GT': 0, 'GN': 0,
        'TA': 0, 'TC': 0, 'TG': 0, 'TT': 0, 'TN': 0,
        'NA': 0, 'NC': 0, 'NG': 0, 'NT': 0, 'NN': 0
    }
    
    # Iterate through the sequence and count dinucleotides
    for i in range(len(seq) - 1):
        dinuc = seq[i:i+2]
        if dinuc in dinuc_d:
            dinuc_d[dinuc] += 1
    
    return dinuc_d


# Optimize memory usage
def reduce_memory_usage(df):
    """ Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage. """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            if df[col].nunique() < df[col].count() / 2:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')

    return df

    ## constructed DataLoaders from Datasets.

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Returns a padded batch of sequences and labels as Tensors.
    """
    seqs, labels = zip(*batch)
    seqs = torch.stack(seqs)  # Stack sequences into a single Tensor
    labels = torch.stack(labels)  # Stack labels into a single Tensor
    return seqs, labels


def build_dataloaders(train_df,
                    val_df,
                    seq_col='seq',
                    target_col='count',
                    one_hot=True,
                    shuffle=True,
                    **kwargs):
    '''
    Given a train and test df with some batch construction
    details, put them into custom SeqDatasetOHE() objects. 
    Give the Datasets to the DataLoaders and return.
    '''

    if one_hot:
        # create Datasets
        train_ds = SeqDatasetOHE(train_df,seq_col=seq_col,target_col=target_col)
        val_ds = SeqDatasetOHE(val_df,seq_col=seq_col,target_col=target_col)

    else:
        # create Datasets
        train_ds = SeqDatasetDNE(train_df,seq_col=seq_col,target_col=target_col)
        val_ds = SeqDatasetDNE(val_df,seq_col=seq_col,target_col=target_col)

    # Put DataSets into DataLoaders
    train_dl = DataLoader(train_ds, shuffle=shuffle, **kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **kwargs,)
    return train_dl,val_dl

def build_dataloader_test(test_df,
                    seq_col='seq',
                    target_col='count',
                    batch_size=128,
                    shuffle=True,
                    one_hot=True,
                    num_workers=24,
                    ):
    '''
    Given a test df with some batch construction
    details, put them into custom SeqDatasetOHE() objects. 
    Give the Datasets to the DataLoaders and return.
    '''

    if one_hot:
        # create Datasets
        test_ds = SeqDatasetOHE(test_df,seq_col=seq_col,target_col=target_col)

    else:
        # create Datasets
        test_ds = SeqDatasetDNE(test_df,seq_col=seq_col,target_col=target_col)

    # Put DataSets into DataLoaders
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return test_dl


#can probaable be deleted
def one_hot_encode_dataset(df, seq_col='seq', target_col='count'):

    seqs = df[seq_col].values
    labels = df[target_col].values

    # One-hot encode sequences and convert to a torch tensor
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    indices = torch.tensor([[nuc_to_idx[nuc] for nuc in seq] for seq in seqs], dtype=torch.long)
    # 0: A, 1: C, 2: G, 3: T, 4: N
    ohe_seqs = F.one_hot(indices, num_classes=5).float()

    # Remove the base N
    mask = indices != 4
    ohe_seqs = ohe_seqs[mask].view(-1, 5)

    # Convert labels to a torch tensor and add an extra dimension
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    #return the one hot encoded sequences and labels
    return ohe_seqs, labels