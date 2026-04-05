#splits the data into train, val and test sets and saves them as tensors

import os
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def one_hot_encode_dataset_fast(df, seq_col='seq', target_col='count'):
    """
    One-hot encode the sequences in the dataframe and convert
    the labels to a torch tensor.
    """
    # One-hot encode sequences and convert to a torch tensor
    # 0: A, 1: C, 2: G, 3: T
    indices = torch.tensor([[ord(c) - ord('A') for c in seq] for seq in df[seq_col]])
    ohe_seqs = F.one_hot(indices, num_classes=4).to_sparse()
    labels = torch.tensor(df[target_col].values)
    return ohe_seqs, labels

def one_hot_encode_dataset(df, seq_col='seq', target_col='count'):
    seqs = df[seq_col].values
    labels = df[target_col].values

    # One-hot encode sequences and convert to a torch tensor
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    indices = torch.tensor([[nuc_to_idx[nuc] for nuc in seq] for seq in seqs], dtype=torch.long)
    # 0: A, 1: C, 2: G, 3: T, 4: N
    ohe_seqs = F.one_hot(indices, num_classes=4).float()

    # Convert labels to a torch tensor and add an extra dimension
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Ensure sequences and labels have the same length
    assert ohe_seqs.size(0) == labels.size(0), "Size mismatch between sequences and labels"

    #switch last two dimensions
    ohe_seqs = ohe_seqs.permute(0,2,1)

    #return the one hot encoded sequences and labels
    return ohe_seqs, labels


def add_reverse_complement(ohe_seqs, labels):
    """
    Add reverse complement to the one-hot encoded sequences
    """
    # Reverse the sequences
    rev_comp = torch.flip(ohe_seqs, [2])
    # complement the sequences A->T, C->G, G->C, T->A
    rev_comp = torch.index_select(rev_comp, 1, torch.tensor([3, 2, 1, 0], dtype=torch.long))
    # Concatenate the original and reverse complement sequences
    ohe_seqs = torch.cat((ohe_seqs, rev_comp), dim=0)
    # Concatenate the original labels with the same labels
    labels = torch.cat((labels, labels), dim=0)
    # Ensure the concatenated sequences and labels have the same length
    assert ohe_seqs.size(0) == labels.size(0), "Size mismatch between sequences and labels"
    return ohe_seqs, labels

def add_context(seq):
    """
    Add context to the sequence
    """

    context = "TCGTCGGCAGCGTCAGATGTGTATCACTGCNNNNNNNNNNNNNNTTGATCACCGCTCCGACTGCAGAAAA"
    context = "CACTGCNNNNNNNNNNNNNNTTGA"

    left = "CACTGC"
    right = "TTGA"

    seq = left + seq + right

    # check if the sequence is valid
    allowed = set("ACGT")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(f"Sequence contains chars not in allowed DNA alphabet (ACGT): {invalid}")

    # ensure the new sequence has the same length as the context
    assert len(seq) == len(context), "Size mismatch between sequence and context"

    return seq

def make_folders(output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Create subdirectories for train, val, and test
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, required=True)
    argparser.add_argument('--output', type=str, required=True)
    argparser.add_argument('--rc', default=False, action='store_true', help='Add reverse complement to the sequence')
    argparser.add_argument('--context', default=False, action='store_true', help='Add context to the sequence')
    argparser.add_argument('--target', type=str, required=True, choices=['log2FoldChange','prop_score','subtraction', 'log2FC'], help='target column for the true label')
    argparser.add_argument('--sequence', type=str, required=True, help='target column for the sequence')

    args = argparser.parse_args()

    df = pd.read_csv(args.data, sep=",")
    if df.columns[0] == 'Unnamed: 0':
        df = df.rename(columns={df.columns[0]: 'seq'})
    print("Data loaded")

    if args.context:
        df[args.sequence] = df[args.sequence].apply(add_context)

    #split data into train and test and validation
    train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    train, val = train_test_split(train, test_size=0.2, shuffle=True, random_state=42)

    make_folders(args.output)
    data_sets = {"train":train,"test":test,"val":val}
    #save the data sets as tensors
    for k,v in data_sets.items():
        seqs,labels = one_hot_encode_dataset(v, seq_col=args.sequence, target_col=args.target)
        if args.rc:
            seqs,labels = add_reverse_complement(seqs, labels)
        #save the one hot encoded sequences and labels
        torch.save(seqs, args.output+f'/{k}/seqs.pt')
        torch.save(labels, args.output+f'/{k}/labels.pt')
    