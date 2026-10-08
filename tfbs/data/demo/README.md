# Bundled demo data

Three small files so that a fresh install can try the data tools without downloading a genome. `tfbs-mcp --setup --demo` registers the genome and the motif with Claude and prints the path of the BED file.

| File | What it is |
|---|---|
| `demo_genome.fa` and `demo_genome.fa.fai` | one record, `demo_chr8`: 10,001 bp of human chromosome 8 around the thyroglobulin (TG) promoter, with its pyfaidx index |
| `demo_peaks.bed` | four 200 bp intervals centred on the strongest NKX2-1 motif matches in that slice |
| `MA1994.1.meme` | the JASPAR NKX2-1 motif in MEME format |

## Provenance

- Assembly: GRCh38.p14, RefSeq accession GCF_000001405.40, downloaded from NCBI Datasets (<https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.40/>).
- Region: record `NC_000008.11` (chromosome 8), positions 132,861,953 to 132,871,953, 1-based and inclusive. The record was renamed `demo_chr8`; the sequence is unchanged (uppercase).
- Landmark: the TG transcription start (Ensembl gene ENSG00000042832, plus strand, GRCh38 start 132,866,953) is at position 5,001 of the slice, so positions 1 to 5,000 are upstream of the gene.
- Peaks: every 13 bp window on both strands was scored with log2-odds from the MA1994.1 letter-probability matrix against the background frequencies in the same file (pseudocount 0.01). The four highest non-overlapping hits at least 101 bp from either end were kept, and each became a 200 bp interval centred on the hit. Coordinates are relative to the slice, 0-based and half-open, as `tfbs_extract_loci` expects. Columns: chrom, start, end, name, log-odds score, strand of the hit.

| Hit | Slice position (1-based) | Strand | Matching sequence on that strand | Log-odds |
|---|---|---|---|---|
| 1 | 405 | minus | AGAACTTGAACTT | 10.37 |
| 2 | 825 | plus | CTCACTTGACCTT | 12.06 |
| 3 | 3,189 | plus | GACACTTCAAATG | 10.36 |
| 4 | 5,153 | minus | GACACTGGAGCTC | 8.99 |

Hit 2 contains the canonical NKX2-1 core `CACTTGA` and lies about 4.2 kb upstream of the transcription start.

- Motif: JASPAR MA1994.1 (Nkx2-1), from <https://jaspar.elixir.no/api/v1/matrix/MA1994.1.meme>. JASPAR is licensed under the Creative Commons Attribution 4.0 International License (<https://creativecommons.org/licenses/by/4.0/>). The file is redistributed unchanged.

## Regenerating

`scripts/make_demo_data.py` in the repository rebuilds this directory from any FASTA that contains the chromosome 8 record (`pyfaidx` is required):

```bash
python scripts/make_demo_data.py --fasta chr8.fa --record NC_000008.11 \
    --start 132861953 --end 132871953 --name demo_chr8 --source GCF_000001405.40 \
    --meme MA1994.1.meme --out tfbs/data/demo \
    --description "human TG (thyroglobulin) promoter region, GRCh38.p14; TG TSS (Ensembl ENSG00000042832, + strand) at position 5001"
```
