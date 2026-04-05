import subprocess
import pandas as pd
from tangermeme.io import extract_loci
import argparse

# Convert the first sequence in X (one-hot encoded) to bases
def one_hot_to_seq(one_hot_seq):
    bases = ['A', 'C', 'G', 'T']
    return ''.join([bases[int(i)] for i in one_hot_seq.argmax(axis=0)])

def write_fasta(X, filename):
    with open(filename, "w") as f:
        for i in range(X.shape[0]):
            f.write(">seq{}\n".format(i))
            f.write(one_hot_to_seq(X[i]) + "\n")


def FIMO_analysis(tf, window=200):
    peaks = pd.read_csv("/sc-projects/sc-proj-cc17-P09_TFBS/data/chipSeq/High_Quality/Oth.ALL.50.{}.AllCell_no_overlaps.bed".format(tf), sep="\t", usecols=(0, 1, 2), names=['chrom', 'start', 'end'])
    X = extract_loci(peaks, "/sc-projects/sc-proj-btg/P09/data/genomes/hg38/hg38.fa", in_window=window, verbose=False).float()
    X = X[X.sum(dim=(1, 2)) == X.shape[-1]]

    write_fasta(X, "/sc-projects/sc-proj-cc17-P09_TFBS/results/fastas/{}/{}.fasta".format(window,tf))

    # Run FIMO using the system installation
    print("Running FIMO for TF:", tf)
    fimo_cmd = [
        "fimo",
        "--thresh", "1",
        "--oc", "/sc-projects/sc-proj-cc17-P09_TFBS/results/FIMO/{}/{}".format(window,tf),
        "--max-strand",
        "--max-stored-scores", "2147483646", # Max int -1 value to avoid memory issues
        "/sc-projects/sc-proj-cc17-P09_TFBS/data/JASPAR/MA1994.1.meme",
        "/sc-projects/sc-proj-cc17-P09_TFBS/results/fastas/{}/{}.fasta".format(window,tf)
    ]
    result = subprocess.run(fimo_cmd, capture_output=True, text=True)

    # Check output and errors
    print(result.stdout)
    print(result.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FIMO analysis")
    parser.add_argument("--tf", type=str, required=True, help="Transcription factor name")
    parser.add_argument("--window", type=int, default=200, help="Window size for locus extraction")
    args = parser.parse_args()
    FIMO_analysis(args.tf, args.window)