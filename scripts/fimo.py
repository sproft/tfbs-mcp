import subprocess
import pandas as pd
from tangermeme.io import extract_loci
import argparse
from pathlib import Path


def one_hot_to_seq(one_hot_seq):
    """Convert a one-hot encoded sequence to a DNA string."""
    bases = ['A', 'C', 'G', 'T']
    return ''.join([bases[int(i)] for i in one_hot_seq.argmax(axis=0)])


def write_fasta(X, filename):
    """Write one-hot encoded sequences to a FASTA file."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        for i in range(X.shape[0]):
            f.write(f">seq{i}\n")
            f.write(one_hot_to_seq(X[i]) + "\n")


def FIMO_analysis(tf, peaks_dir, genome_fasta, results_dir, motif_file, window=200):
    """Run FIMO motif scanning on extracted ChIP-seq peak sequences."""
    peak_file = Path(peaks_dir) / f"Oth.ALL.50.{tf}.AllCell_no_overlaps.bed"
    peaks = pd.read_csv(peak_file, sep="\t", usecols=(0, 1, 2), names=['chrom', 'start', 'end'])
    X = extract_loci(peaks, genome_fasta, in_window=window, verbose=False).float()
    X = X[X.sum(dim=(1, 2)) == X.shape[-1]]

    fasta_path = Path(results_dir) / "fastas" / str(window) / f"{tf}.fasta"
    write_fasta(X, str(fasta_path))

    fimo_output = Path(results_dir) / "FIMO" / str(window) / tf
    print("Running FIMO for TF:", tf)
    fimo_cmd = [
        "fimo",
        "--thresh", "1",
        "--oc", str(fimo_output),
        "--max-strand",
        "--max-stored-scores", "2147483646",
        str(motif_file),
        str(fasta_path),
    ]
    result = subprocess.run(fimo_cmd, capture_output=True, text=True)

    print(result.stdout)
    print(result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FIMO analysis on ChIP-seq peaks")
    parser.add_argument("--tf", type=str, required=True, help="Transcription factor name")
    parser.add_argument("--window", type=int, default=200, help="Window size for locus extraction")
    parser.add_argument("--peaks-dir", type=str, required=True, help="Directory containing peak BED files")
    parser.add_argument("--genome-fasta", type=str, required=True, help="Path to reference genome FASTA")
    parser.add_argument("--results-dir", type=str, required=True, help="Base directory for results output")
    parser.add_argument("--motif-file", type=str, required=True, help="Path to MEME-format motif file")
    args = parser.parse_args()
    FIMO_analysis(args.tf, args.peaks_dir, args.genome_fasta, args.results_dir, args.motif_file, args.window)
