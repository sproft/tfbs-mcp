"""Build the bundled demo data in tfbs/data/demo/ from a GRCh38 FASTA.

The demo lets a fresh install exercise tfbs_extract_loci and tfbs_fimo without
downloading a genome: a short slice of chromosome 8 around the thyroglobulin
(TG) transcription start, a BED of the strongest NKX2-1 motif matches in that
slice, and the JASPAR MA1994.1 motif.

Usage::

    python scripts/make_demo_data.py --fasta chr8.fa --record NC_000008.11 \
        --start 132861953 --end 132871953 --meme MA1994.1.meme --out tfbs/data/demo

Coordinates are 1-based inclusive on the source record. The output record is
renamed (default demo_chr8) and the BED is written in slice coordinates.
"""

from __future__ import annotations

import argparse
import math
import shutil
import textwrap
from pathlib import Path

from pyfaidx import Fasta

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def read_meme(path: Path) -> tuple[str, list[dict[str, float]], dict[str, float]]:
    """Return (motif id, per-position probabilities, background) from a MEME file."""
    lines = path.read_text().splitlines()
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    motif_id, matrix, in_matrix = "", [], False
    for i, line in enumerate(lines):
        if line.startswith("Background letter frequencies"):
            fields = lines[i + 1].split()
            background = {fields[j]: float(fields[j + 1]) for j in range(0, len(fields), 2)}
        elif line.startswith("MOTIF"):
            motif_id = line.split()[1]
        elif line.startswith("letter-probability matrix"):
            in_matrix = True
        elif in_matrix:
            fields = line.split()
            if len(fields) != 4:
                in_matrix = False
                continue
            matrix.append(dict(zip("ACGT", map(float, fields))))
    if not matrix:
        raise SystemExit(f"no letter-probability matrix found in {path}")
    return motif_id, matrix, background


def log_odds(matrix, background, pseudocount: float = 0.01):
    return [{b: math.log2((row[b] + pseudocount) / (background[b] + pseudocount)) for b in "ACGT"}
            for row in matrix]


def score(window: str, lo) -> float:
    return sum(col[base] for col, base in zip(lo, window))


def best_matches(seq: str, lo, n: int, min_distance: int, margin: int):
    """Top-n non-overlapping motif hits on either strand, at least `margin` bp from the ends."""
    w = len(lo)
    hits = []
    for i in range(margin, len(seq) - w - margin):
        window = seq[i:i + w]
        if "N" in window:
            continue
        rc = window.translate(COMPLEMENT)[::-1]
        fwd, rev = score(window, lo), score(rc, lo)
        if fwd >= rev:
            hits.append((fwd, i, "+", window))
        else:
            hits.append((rev, i, "-", rc))
    hits.sort(reverse=True)
    chosen = []
    for hit in hits:
        if all(abs(hit[1] - c[1]) >= min_distance for c in chosen):
            chosen.append(hit)
        if len(chosen) == n:
            break
    return sorted(chosen, key=lambda h: h[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True, type=Path, help="FASTA holding the source record")
    ap.add_argument("--record", required=True, help="record name in --fasta, e.g. NC_000008.11")
    ap.add_argument("--start", required=True, type=int, help="1-based inclusive start on the record")
    ap.add_argument("--end", required=True, type=int, help="1-based inclusive end on the record")
    ap.add_argument("--name", default="demo_chr8", help="name of the output record")
    ap.add_argument("--meme", required=True, type=Path, help="MEME motif file to scan with and to copy")
    ap.add_argument("--out", required=True, type=Path, help="output directory (tfbs/data/demo)")
    ap.add_argument("--n-peaks", type=int, default=4)
    ap.add_argument("--window", type=int, default=200, help="BED interval width around each hit")
    ap.add_argument("--description", default="", help="free text appended to the FASTA header")
    ap.add_argument("--source", default="",
                    help="assembly accession named in the FASTA header, e.g. GCF_000001405.40 "
                         "(default: the --fasta file name)")
    args = ap.parse_args()

    genome = Fasta(str(args.fasta))
    seq = str(genome[args.record][args.start - 1:args.end]).upper()
    expected = args.end - args.start + 1
    if len(seq) != expected:
        raise SystemExit(f"got {len(seq)} bp, expected {expected}; is the region inside the record?")

    args.out.mkdir(parents=True, exist_ok=True)
    fa_path = args.out / "demo_genome.fa"
    source = f"{args.source or args.fasta.name}:{args.record}:{args.start}-{args.end}"
    header = f">{args.name} {source}" + (f" {args.description}" if args.description else "")
    # The .fai records byte offsets and the line width, so the endings must be
    # LF on every platform or the index is wrong wherever the file is checked out.
    with open(fa_path, "w", newline="\n") as fh:
        fh.write(header + "\n" + "\n".join(textwrap.wrap(seq, 80)) + "\n")
    for stale in (fa_path.with_suffix(".fa.fai"), Path(str(fa_path) + ".fai")):
        if stale.exists():
            stale.unlink()
    Fasta(str(fa_path))  # writes demo_genome.fa.fai next to the file
    # pyfaidx writes the index in text mode, so normalise it to LF as well.
    fai_path = Path(str(fa_path) + ".fai")
    fai_path.write_bytes(fai_path.read_bytes().replace(b"\r\n", b"\n"))

    motif_id, matrix, background = read_meme(args.meme)
    lo = log_odds(matrix, background)
    half = args.window // 2
    hits = best_matches(seq, lo, args.n_peaks, min_distance=args.window, margin=half + 1)
    if len(hits) < args.n_peaks:
        raise SystemExit(f"only {len(hits)} non-overlapping hits found")

    bed_lines = []
    for k, (s, pos, strand, window) in enumerate(hits, 1):
        centre = pos + len(lo) // 2
        bed_lines.append("\t".join([args.name, str(centre - half), str(centre + half),
                                    f"{motif_id}_hit_{k}", f"{s:.2f}", strand]))
    with open(args.out / "demo_peaks.bed", "w", newline="\n") as fh:
        fh.write("\n".join(bed_lines) + "\n")

    shutil.copyfile(args.meme, args.out / args.meme.name)

    gc = (seq.count("G") + seq.count("C")) / len(seq)
    print(f"wrote {fa_path} ({len(seq):,} bp, GC {gc:.1%}) and its .fai")
    print(f"wrote demo_peaks.bed with {len(hits)} {motif_id} hits:")
    for k, (s, pos, strand, window) in enumerate(hits, 1):
        print(f"  hit {k}: position {pos + 1} ({strand}) {window}  log-odds {s:.2f}")
    print(f"copied {args.meme.name}")


if __name__ == "__main__":
    main()
