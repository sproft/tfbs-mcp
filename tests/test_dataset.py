# create artifical testdateset csv that contains random sequences and log2FC values half the sequences will contain a motif or the reverse complement motif in a random position
import os
import pandas as pd
import numpy as np
import torch



def create_artificial_tfbs_dataset(file_path, num_samples=1048313, seq_length=14, motif="CACTTGA", motif_insertion_rate=0.5):
    nucleotides = ['A', 'C', 'G', 'T']
    sequences = []
    log2fc_values = []

    motif_length = len(motif)
    rev_comp_motif = motif.translate(str.maketrans('ACGT', 'TGCA'))[::-1]

    for _ in range(num_samples):
        seq = ''.join(np.random.choice(nucleotides, seq_length))
        
        if np.random.rand() < motif_insertion_rate:
            insert_pos = np.random.randint(0, seq_length - motif_length + 1)
            if np.random.rand() < 0.5:
                seq = seq[:insert_pos] + motif + seq[insert_pos + motif_length:]
            else:
                seq = seq[:insert_pos] + rev_comp_motif + seq[insert_pos + motif_length:]
            log2fc = np.random.uniform(1.0, 3.0)  # Higher log2FC for sequences with motif
        else:
            log2fc = np.random.uniform(-1.0, 1.0)  # Lower log2FC for sequences without motif

        sequences.append(seq)
        log2fc_values.append(log2fc)

    df = pd.DataFrame({'seq': sequences, 'log2FC': log2fc_values})
    df.to_csv(file_path, index=False)
    print(f"Artificial TFBS dataset created at: {file_path}")


if __name__ == "__main__":
    create_artificial_tfbs_dataset("../data/test/artificial_dataset.csv", num_samples=1000)

