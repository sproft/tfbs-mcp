# Smoke test of all 33 tools on a SLURM cluster

`docs/all_tools_plan.json` is a plan of 49 tool calls that calls each of the 33 tools of the server at least once, on real NKX2-1 data, in about 50 minutes on one GPU node. `scripts/record_demo_session.py` plays it: it starts `tfbs-mcp` over the stdio protocol as an AI app does, sends the calls in order, takes later arguments from earlier results (`$ref`), and writes every argument, result and wall time to one JSON log. No LLM is involved, so a failure in the log is a failure of the server, its pipelines or the environment, never of a client's choice of arguments.

This page says what to stage before the run, the command, what each phase proves, which steps are expected to fail where, and how to read the log. It was written for the BIH cluster (GPU partition `gpu`, the NKX2-1 project data under `/sc-projects/sc-proj-cc17-P09_TFBS`); the paths below are the ones that the 2026-10-13 recording session reported, so check them before you use them.

## 1. Placeholders

The plan names its inputs as `@KEY` placeholders, which the recorder replaces with `--set KEY=VALUE` before it sends anything.

| Placeholder | What it is | Value on the BIH cluster |
|---|---|---|
| `@PROJECT` | project root of the server, also passed as `TFBS_PROJECT_ROOT`; training writes `saved_models/` and the analysis tools write `results/` here | `$T/project` |
| `@DATA` | directory holding `nkx2-1_chipatlas.bed`, `nkx2-1_remap2022.bed` and `chipatlas_intersect/` | `$T/data` |
| `@GENOME` | hg38 FASTA with its `.fai` (UCSC names: `chr8`) | `/sc-scratch/sc-scratch-btg/genomes/hg38/hg38.fa` |
| `@MOTIF` | JASPAR MA1994.1 in MEME format | `/sc-projects/sc-proj-cc17-P09_TFBS/data/JASPAR/MA1994.1.meme` |
| `@CLINVAR` | GRCh38 `clinvar.vcf.gz` with its `.tbi` next to it | `/sc-projects/sc-proj-cc17-P09_TFBS/data/clinvar/clinvar.vcf.gz` |
| `@PDB` | the repaired NKX2-1 TF-DNA complex | `/sc-projects/sc-proj-cc17-P09_TFBS/data/structures/repaired/NKX2-1_complex_CAB_Repair.pdb` |
| `@PARTITION` | GPU partition, also passed as `TFBS_SLURM_PARTITION` | `gpu` |
| `@WORK` | scratch directory for every file the plan names itself | `$T/work` |
| `@RELEASE` | directory that contains `models/flank/vcnnbpnet.pt` of the released NKX2-1 models (doi:10.5281/zenodo.21399578) | see 2.6 |
| `@SCRIPT` | a trivial sbatch script for `slurm_submit` | `$T/smoke_job.sh` |

`$T` is the test folder, `T=$(realpath ~/P09_new/tfbs-mcp-testruns)`. Use the resolved path, since the jobs the plan submits read it on other nodes.

The four reference files are passed twice: as `--set` values for the steps that name them, and as the server variables `TFBS_GENOME_FASTA`, `TFBS_MOTIF_FILE`, `TFBS_CLINVAR_VCF` and `TFBS_FOLDX_PDB` for the steps that leave them out (steps 2, 8, 9 and 35 to 38). Genome, motif and PDB are tested both ways: by argument in steps 7, 28, 31, 33 and 34, from the environment in steps 2, 8, 9 and 35 to 38. ClinVar is tested only through the argument: step 38, the only ClinVar step, passes `clinvar_vcf=@CLINVAR`, so the tool's fallback to `TFBS_CLINVAR_VCF` is not exercised, and the variable is only checked by `tfbs_server_info` (step 0, `exists` true).

## 2. Before the allocation

Everything in this section can be done while you wait for the GPU. None of it needs a GPU; the conversion in 2.3 is light, but run it in a small CPU allocation if the login node rules ask for that.

### 2.1 Install the newest server

On a node that reaches GitHub:

```bash
conda activate tfbs-mcp
pip install --force-reinstall --no-deps "tfbs-mcp[mcp,genomics,viz,foldx] @ git+https://github.com/sproft/tfbs-mcp.git"
tfbs-mcp --check                  # 33 tools, no FAIL line; drop --no-deps above if it reports a missing package
pip freeze | grep tfbs-mcp        # the commit after "@" is the version under test
```

`--check` notes that `wandb` is missing; that is expected (step 15, section 5).

### 2.2 The recorder, the plan and the folders

The recorder and the plan are not installed with the package. Take them from a clone at the same commit, or copy `scripts/record_demo_session.py` and `docs/all_tools_plan.json` from the branch that holds them:

```bash
T=$(realpath ~/P09_new/tfbs-mcp-testruns)
git clone https://github.com/sproft/tfbs-mcp.git "$T/tfbs-mcp"      # then check out the branch with docs/all_tools_plan.json
mkdir -p "$T/project" "$T/data" "$T/work" "$T/release"
cp /sc-projects/sc-proj-cc17-P09_TFBS/tfbs-mcp-rec/data/nkx2-1_chipatlas.bed "$T/data/"
cp /sc-projects/sc-proj-cc17-P09_TFBS/tfbs-mcp-rec/data/nkx2-1_remap2022.bed "$T/data/"
```

The ChIP-Atlas BED is the 9-column ChIP-Atlas file the 2026-10-13 session trained on (34,700 peaks, MACS2 score in column 5, experiment metadata in column 4). The ReMap BED is the ReMap 2022 non-redundant NKX2-1 file (124,382 peaks, experiment count in column 5; 105 of them have a count of at least 4).

### 2.3 The ChIP-Atlas benchmark inputs

`tfbs_chipseq_benchmark` (step 35) and `tfbs_chipseq_method_comparison` (step 36) read one file per TF from one directory, `intersect_dir`, named exactly `<TF>.allintersect.bed`, with `<TF>` the name passed as `pos_tf` or in `neg_tfs`. Both pipelines (`tfbs/pipelines/chipseq_benchmark.py` and `chipseq_benchmark_with_foldx.py`) read each file with pandas as tab-separated, take the first line as a header and skip it, and use only the first four columns as chrom, start, end and `num`, the number of experiments that support the interval, which must be an integer on every line. Further columns are ignored. Overlapping intervals are merged, keeping the largest `num`. The benchmark uses `num` for its experiment-count cutoffs and its regression; the method comparison uses the intervals only.

The plan needs these files in `$T/data/chipatlas_intersect/`:

| File | Needed by | When missing |
|---|---|---|
| `NKX2-1.allintersect.bed` | steps 35 and 36 | both steps fail |
| `GATA1.allintersect.bed` | steps 35 and 36 | step 35 skips GATA1; step 36 fails, because it then has no negative TF at all |
| `MYOD1`, `NKX2-5`, `RXRA` `.allintersect.bed` | step 35 | step 35 prints "Skipping" and goes on |

A ChIP-Atlas `Oth.ALL.50.<TF>.AllCell.bed` file is not in this layout. It has one line per peak, each called in one experiment, no column header (it may start with a `track` line, which the recipe below removes), the experiment metadata (`ID=SRX...;Name=...`) in column 4 and the MACS2 score in column 5, so the experiment count per region is not in it, and column 4 is not a number. It does hold everything needed to make the file, though: split it by experiment ID and count the overlaps with `bedtools multiinter -header`, which writes a header line and then chrom, start, end, the number of experiments and the list of them, the layout the README describes. So the AllCell files are enough together with bedtools; nothing else has to be downloaded.

First check whether the converted files already exist. The thesis notes list `chipSeq/Markus/INTERSECTIONS/` with intersection BEDs for NKX2-1, GATA1, MYOD1, NKX2-5, RXRA and FOXA1, and the project's TAD scripts read `*.allintersect.bed` from there:

```bash
I=/sc-projects/sc-proj-cc17-P09_TFBS/data/chipSeq/Markus/INTERSECTIONS
for TF in NKX2-1 GATA1 MYOD1 NKX2-5 RXRA; do head -2 "$I/$TF.allintersect.bed"; done
```

Each file should show a header line, then a line whose fourth column is an integer. If so, link the directory: `ln -s "$I" "$T/data/chipatlas_intersect"`. Otherwise build the files from the AllCell files (the directory below is the one the project's prediction scripts use; adjust it), with bedtools from a module or conda:

```bash
SRC=/sc-projects/sc-proj-cc17-P09_TFBS/data/chipSeq/High_Quality
OUT="$T/data/chipatlas_intersect"; mkdir -p "$OUT"
for TF in NKX2-1 GATA1 MYOD1 NKX2-5 RXRA; do
  tmp=$(mktemp -d)
  grep -v '^track' "$SRC/Oth.ALL.50.$TF.AllCell.bed" \
    | awk -F'\t' -v d="$tmp" '{split($4, a, ";"); id = substr(a[1], 4); print $1 "\t" $2 "\t" $3 > (d "/" id ".bed")}'
  for f in "$tmp"/*.bed; do sort -k1,1 -k2,2n "$f" -o "$f"; done
  bedtools multiinter -header -i "$tmp"/*.bed > "$OUT/$TF.allintersect.bed"
  rm -r "$tmp"
done
head -2 "$OUT"/*.allintersect.bed
```

Then check the size of what you staged, whichever way you got it:

```bash
wc -l "$T"/data/chipatlas_intersect/*.allintersect.bed
ls -lhL "$T"/data/chipatlas_intersect/*.allintersect.bed
```

The size matters for the run time, and `max_sequences` (step 35) and `n_per_tf` (step 36) do not bound it: they cap only the windows that are scored. Both pipelines read every staged file in full and merge its intervals row by row, which ran at about 28,000 lines per second on the workstation (300,000 lines in 10.8 s with the pipeline's own function). Step 35 merges the NKX2-1 file once and each negative TF file twice; step 36 merges NKX2-1 and GATA1 once each. So the two steps spend about (2 x NKX2-1 + 3 x GATA1 + 2 x (MYOD1 + NKX2-5 + RXRA)) / 28,000 seconds on merging, with each TF counted by its number of lines. A `bedtools multiinter` file of a TF with many experiments has one line per breakpoint, so it can run to millions of lines. If the sum comes to more than a few minutes, cut it in one of two ways:

- Stage only NKX2-1 and GATA1. MYOD1, NKX2-5 and RXRA are optional: step 35 prints "Skipping" for a missing file and goes on. If you linked the project directory, link the two files instead: `rm "$T/data/chipatlas_intersect"` (this removes only the link), `mkdir "$T/data/chipatlas_intersect"`, `ln -s "$I/NKX2-1.allintersect.bed" "$I/GATA1.allintersect.bed" "$T/data/chipatlas_intersect/"`.
- Pre-merge the files once. The awk below does the merge the pipelines do (overlapping and book-ended intervals joined, the largest `num` kept) and keeps only the four columns they read, so the merged regions, the windows sampled from them and the results do not change. On a synthetic 300,000-line multiinter-style file it left 45,409 lines; the pipelines' merge gave the same regions and the same 2,000-window sample from both files, and reading and merging took 1.7 s instead of 10.8 s. It writes new files and leaves the source directory alone:

```bash
SRCI=$(realpath "$T/data/chipatlas_intersect")
M="$T/data/chipatlas_intersect_merged"; mkdir -p "$M"
for TF in NKX2-1 GATA1 MYOD1 NKX2-5 RXRA; do
  [ -f "$SRCI/$TF.allintersect.bed" ] || continue
  { printf 'chrom\tstart\tend\tnum\n'
    tail -n +2 "$SRCI/$TF.allintersect.bed" | cut -f1-4 | LC_ALL=C sort -k1,1 -k2,2n \
      | awk -F'\t' -v OFS='\t' '$1 != c || $2 + 0 > e { if (c != "") print c, s, e, n; c = $1; s = $2 + 0; e = $3 + 0; n = $4 + 0; next }
          { if ($3 + 0 > e) e = $3 + 0; if ($4 + 0 > n) n = $4 + 0 }
          END { if (c != "") print c, s, e, n }'
  } > "$M/$TF.allintersect.bed"
done
wc -l "$M"/*.allintersect.bed
if [ -L "$T/data/chipatlas_intersect" ]; then rm "$T/data/chipatlas_intersect"; else mv "$T/data/chipatlas_intersect" "$T/data/chipatlas_intersect_unmerged"; fi
mv "$M" "$T/data/chipatlas_intersect"
```

The method comparison cuts 24 bp windows (its pipeline's default, which the tool does not change), so the model it scores must take 24 bp input; the classifier the plan trains does.

### 2.4 The trivial sbatch script

`slurm_submit` (step 43) submits `@SCRIPT` with one argument, the seconds to sleep. Write it as `$T/smoke_job.sh`:

```bash
cat > "$T/smoke_job.sh" <<'EOF'
#!/bin/bash
#SBATCH --job-name=tfbs_smoke
#SBATCH --time=00:05:00
#SBATCH --mem=100M
#SBATCH --cpus-per-task=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
echo "tfbs-mcp smoke job ${SLURM_JOB_ID} on $(hostname)"
echo "stderr line" >&2
sleep "${1:-20}"
echo done
EOF
chmod +x "$T/smoke_job.sh"
```

It names no partition, so it goes to the cluster's default partition. Its logs land next to the script, named after the job name the plan passes (`tfbs_smoke_submit_<job id>.out`).

### 2.5 The sweep config

`tfbs_sweep` (step 15) needs an existing sweep YAML. Copy the bundled one into the project:

```bash
cp "$(python -c 'import tfbs, pathlib; print(pathlib.Path(tfbs.__file__).parent / "configs" / "wandb_sweep_config.yaml")')" "$T/project/"
```

### 2.6 The released FLANK model

Steps 17, 23, 24 and 29 load the released FLANK checkpoint: it is the one inference checkpoint (the format training here cannot write) and the one regression model with label scaling in the run, so it tests what the trained classifier cannot. Download and unpack the Zenodo archive, then point `RELEASE` at the folder that holds `models/`:

```bash
cd "$T/release"
curl -L -o nkx2-1-tfbs-variants.zip "https://zenodo.org/api/records/21399578/files/sproft/nkx2-1-tfbs-variants-v1.0.0.zip/content"
unzip -q nkx2-1-tfbs-variants.zip
RELEASE=$(dirname "$(dirname "$(dirname "$(find "$T/release" -path '*models/flank/vcnnbpnet.pt' | head -1)")")")
ls "$RELEASE/models/flank/vcnnbpnet.pt"
```

### 2.7 Checks of the reference files and programs

```bash
which fimo foldx sbatch squeue sacct scancel     # each must print a path (foldx may be foldx5)
G=/sc-scratch/sc-scratch-btg/genomes/hg38/hg38.fa
P=/sc-projects/sc-proj-cc17-P09_TFBS/data/structures/repaired/NKX2-1_complex_CAB_Repair.pdb
ls -l "$G" "$G.fai" /sc-projects/sc-proj-cc17-P09_TFBS/data/JASPAR/MA1994.1.meme \
      /sc-projects/sc-proj-cc17-P09_TFBS/data/clinvar/clinvar.vcf.gz /sc-projects/sc-proj-cc17-P09_TFBS/data/clinvar/clinvar.vcf.gz.tbi "$P"
python -c "import pyfaidx; print(pyfaidx.Fasta('$G')['chr8'][132862769:132862793])"
awk '/^ATOM/ {print substr($0, 22, 1), substr($0, 18, 3)}' "$P" | sort -u | awk '{c[$1] = c[$1] " " $2} END {for (k in c) print k ":" c[k]}'
env | grep -i _proxy
```

- The genome slice must read `TGGGCTCCTCACTTGACCTTTGCT` (upper or lower case). It is the 24 bp window of the variant chr8:132862782 T>G that step 7 extracts; it equals position 830 of the bundled demo genome, so a different slice means another genome build and step 7 reports a `ref_mismatch`.
- The chain listing must show protein residues (ALA, ARG, ...) on chain A and DNA residues on chains C and D (in the repaired file they are named A, C, G, T, not DA, DC, DG, DT): step 33 passes `chains_protein="A"`, `chains_dna="C,D"`, the chain IDs of the repaired NKX2-1 complex, which differ from those of PDB entry 9U19 (protein C, DNA A and B). If the file uses other IDs, change those two values in step 33 of the plan.
- If the proxy listing prints anything, step 32 needs it to reach RCSB; the command in section 3 passes the usual proxy variables on, because the recorder starts the server with a minimal environment.

## 3. The run

The recorder starts the server with the MCP SDK's default environment (on Linux only `HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM` and `USER`) plus the `--env` values. `fimo` and `foldx` are found through `PATH`. The SLURM job variables, `CUDA_VISIBLE_DEVICES` and proxy settings are not inherited, so the command passes on those that are set (an empty `CUDA_VISIBLE_DEVICES` would hide the GPU, hence the test for a value), which makes `tfbs_server_info` report the allocation, keeps the server on the allocated GPU and lets step 32 use the proxy. `LD_LIBRARY_PATH` is not passed, which also avoids the library conflict the 2026-10-12 session hit. The jobs the server submits inherit the server's environment, so they do not carry the allocation's memory settings either.

Take the allocation in tmux on the login node. The plan runs up to 24 FoldX processes at once (`foldx_workers` 24 in steps 36 to 38), so ask for at least 25 CPUs; with fewer, those steps only take longer.

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=32 --mem=64G --time=02:00:00 --pty bash -i
```

Inside the allocation:

```bash
conda activate tfbs-mcp
T=$(realpath ~/P09_new/tfbs-mcp-testruns); cd "$T"
RELEASE=<the folder from 2.6>
COMMIT=$(pip freeze | sed -n 's/.*tfbs-mcp\.git@\([0-9a-f]\{7\}\).*/\1/p')
RUN=$(date +%Y%m%d-%H%M%S)
EXTRA=()
for v in SLURM_JOB_ID SLURM_JOB_PARTITION SLURM_GPUS_ON_NODE CUDA_VISIBLE_DEVICES \
         http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
  [ -n "${!v}" ] && EXTRA+=(--env "$v=${!v}")
done
echo "${EXTRA[@]}"                # expect SLURM_JOB_ID and a GPU variable
python "$T/tfbs-mcp/scripts/record_demo_session.py" --server "$(which tfbs-mcp)" \
    --plan "$T/tfbs-mcp/docs/all_tools_plan.json" --out "$T/session_log_$RUN.json" \
    --commit "$COMMIT" --timeout 2400 \
    --env TFBS_PROJECT_ROOT="$T/project" --env TFBS_CALL_LOG="$T/calls_$RUN.jsonl" \
    --env TFBS_SLURM_PARTITION=gpu \
    --env TFBS_GENOME_FASTA=/sc-scratch/sc-scratch-btg/genomes/hg38/hg38.fa \
    --env TFBS_MOTIF_FILE=/sc-projects/sc-proj-cc17-P09_TFBS/data/JASPAR/MA1994.1.meme \
    --env TFBS_CLINVAR_VCF=/sc-projects/sc-proj-cc17-P09_TFBS/data/clinvar/clinvar.vcf.gz \
    --env TFBS_FOLDX_PDB=/sc-projects/sc-proj-cc17-P09_TFBS/data/structures/repaired/NKX2-1_complex_CAB_Repair.pdb \
    "${EXTRA[@]}" \
    --set PROJECT="$T/project" --set DATA="$T/data" --set WORK="$T/work" \
    --set GENOME=/sc-scratch/sc-scratch-btg/genomes/hg38/hg38.fa \
    --set MOTIF=/sc-projects/sc-proj-cc17-P09_TFBS/data/JASPAR/MA1994.1.meme \
    --set CLINVAR=/sc-projects/sc-proj-cc17-P09_TFBS/data/clinvar/clinvar.vcf.gz \
    --set PDB=/sc-projects/sc-proj-cc17-P09_TFBS/data/structures/repaired/NKX2-1_complex_CAB_Repair.pdb \
    --set PARTITION=gpu --set RELEASE="$RELEASE" --set SCRIPT="$T/smoke_job.sh"
```

The recorder prints one line per step (`[07] tfbs_extract_loci -> ok (0.4s)`) on stderr and writes the log at the end, also when it crashes. `--timeout 2400` is the limit per call; the longest steps (36 to 38) should take about 5 to 10 minutes each. The server's call log is a new file per run and is copied into the session log.

A rerun overwrites the files under `$T/work` and the checkpoint under `$T/project/saved_models/`. Start from empty folders when you want a clean record.

## 4. What each phase proves

Step numbers are the plan's indices, which are also the indices in the session log and the first number of every `$ref`. Each step's `note` in the plan says what it checks and what a good result looks like.

- **Setup (0, 1).** The server starts over stdio and lists 33 tools (`tools_listed` in the log), sees the GPU and the allocation, takes its project root, call log, default partition and the four reference files from the environment (`reference_files`, each with `exists` true), and holds no model yet.
- **Data prep (2 to 9).** The ChIP-Atlas BED is read with its metadata column intact (the inline `#` fix) and the MACS2 scores of column 5 are kept (`has_scores` true); GC-matched negatives are drawn from hg38; the classification, split and regression datasets are built from the files of the extraction, all through `$ref`; inline sequences are one-hot encoded with reverse complements; a variant window is cut from hg38, which also proves the genome build; two small samples are drawn for scoring and for the ClinVar check.
- **Training (10 to 15).** `tfbs_config` infers a classifier from 0/1 labels and writes the checkpoint path the benchmark tools look up; `tfbs_train` trains on the node's GPU and reports the best epoch's metrics; a second config with explicit paths is submitted as a SLURM GPU job (config mode, with `extra_args`, `cpus_per_task` and `mem`); `tfbs_validate` submits a test-split evaluation using the default partition from `TFBS_SLURM_PARTITION`; `tfbs_sweep` is expected to fail (section 5).
- **Analysis (16 to 31).** Both checkpoint formats load (Lightning and inference); prediction works on `.pt` files, inline sequences and in window mode; ROC-AUC comes from two prediction results; FLANK scores the variant's ref and alt windows in label units (about 1.07 and -0.54 log2 fold change, the values of the same window in the bundled demo); attribution, mutagenesis (on the `tfbs_preprocess` file), motif marginalisation, seqlets with TOMTOM annotation, epistasis, filter PFMs and FIMO (the MEME Suite binary, `backend` `meme`) all run on a handful of sequences.
- **Structure (32 to 34).** A structure is fetched from RCSB (with AlphaFold DB as the fallback), FoldX AnalyseComplex scores the registered repaired complex, and a one-window FoldX scan writes a bigWig.
- **Validation (35 to 38).** Each of the four pipelines runs to the end in direct mode on reduced inputs and returns its parsed results: the NN benchmark finds the trained checkpoint through the `models_dir` lookup (with a warning, since a classifier lies under `none/`, not `standardize/`); the method comparison runs NN, FIMO and FoldX on 20 windows each of NKX2-1 and GATA1; the ReMap validation runs all three methods on 20 peaks with at least 4 experiments against their shuffled copies; the ClinVar check runs all three methods on the variants inside 1,000 sampled peak windows. These steps prove that the pipelines, `fimo`, pysam and the reference files work together. They prove that FoldX works inside them only when the FoldX checks of section 6 pass. Since version 1.0.0 the three pipelines stop with the FoldX-missing error when a structure is set and they find no FoldX binary, so an `ok` step shows that FoldX was found, but not that it ran, because all three turn a failed FoldX run into NaN. At 6fbec08, the commit of the 2026-10-13 run, a step could be `ok` without FoldX even being found: the method comparison then skipped FoldX without an error and the ClinVar check wrote empty (NaN) FoldX columns. Their AUCs are on tens of sequences and say nothing about model quality. The full-size runs took 26 minutes (ReMap, 105 against 105 peaks) and 137 minutes (ClinVar, 645 variants) on 2026-10-13. The reductions come from the tools' own parameters: `max_positives` (ReMap), `n_per_tf` and a single negative TF (method comparison), `max_sequences` (benchmark; it caps the windows scored, not the reading and merging of the intersect files, see 2.3), and a sampled peak BED from `tfbs_extract_loci` (ClinVar, whose BED has no numeric column 4, so `peak_confidence` cannot shrink it).
- **SLURM (39 to 48).** The scheduler tools act on the jobs the run submitted: status and both logs of the training job (submitted about half an hour earlier, so it has usually run unless the GPU queue is long), a monitoring request and the logs of the validation job, a submission of `@SCRIPT` with an argument and an `sbatch` flag, a second submission from `script_body`, its cancellation, and the status after the cancel. A pending job is not an error: `slurm_status` then reports `PENDING` and `slurm_logs` reports the expected log path with a not-found note.

## 5. Expected errors

Only step 15 carries `"expect_error": true`. The recorder does not read `note` or `expect_error`; they document the plan.

| Step | Tool | Environment | Expected answer |
|---|---|---|---|
| 15 | `tfbs_sweep` | `wandb` not installed (the install in 2.1 has no `tracking` extra) | error "tfbs_sweep needs wandb, which is not installed", no sweep created |
| 15 | `tfbs_sweep` | `wandb` installed, no `wandb login` | an error from `wandb sweep` ("wandb sweep failed" with its stderr), no sweep created |
| 15 | `tfbs_sweep` | `wandb` installed and logged in | no error: a one-trial sweep in project `tfbs-mcp-smoke` and one agent job on the GPU partition; cancel it with `scancel` if you do not want it |
| 32 | `tfbs_fetch_structure` | node without outbound HTTPS | error listing both attempts (`pdb_id`, `uniprot_id`); the other steps do not depend on it |
| 39 to 48 | SLURM tools | `sbatch` refused inside the allocation | `slurm_submit` and the submitting steps 13 and 14 return the `sbatch` error, and the steps that take their `job_id` fail with a `$ref` error |

Steps 36 to 38 fail when the pipeline finds no FoldX binary (`TFBS_FOLDX_BIN`, else `foldx` or `foldx5` on the server's PATH): the error starts "FoldX binary not found on PATH. A FoldX structure is set", the result holds the pipeline's `stderr`, and the fix suggests `skip_foldx=true`. None of the three validations fails when single FoldX runs fail: the FoldX values are then missing, which only the checks in section 6 show. At 6fbec08, the commit of the 2026-10-13 run, steps 36 and 38 did not fail without `foldx`, and step 37 failed with "ReMap validation failed (exit 1)".

On a machine without SLURM, FoldX, the `fimo` program or pysam (the local dry run on Windows), steps 13 to 15, 33, 34, 36 to 38 and 39 to 48 fail with the missing program in the error, or with a `$ref` error naming the missing `job_id`; section 8 lists the exact answers.

## 6. Reading the log

Every tool step in `session_log_<run>.json` has `arguments` (after `--set` and `$ref`), `result` (the parsed JSON answer), `seconds`, `seconds_since_launch` and `is_error`, which is true when the call failed, timed out, a `$ref` named a missing key, or the answer has an `"error"` key. A failed `$ref` logs `ref`, `missing_key` and `available_keys`. The server's own call log is under `server_call_log`, so `scripts/session_from_call_log.py` is not needed. One overview:

```bash
python - "$T/session_log_$RUN.json" "$T/tfbs-mcp/docs/all_tools_plan.json" <<'EOF'
import json, sys
log = json.load(open(sys.argv[1])); plan = json.load(open(sys.argv[2]))
print("tools listed:", log["tools_listed"], " startup s:", log["seconds_to_initialize"])
for i, (step, planned) in enumerate(zip(log["steps"], plan)):
    r = step.get("result") if isinstance(step.get("result"), dict) else {}
    flag = "ERROR" if step["is_error"] else "ok"
    if step["is_error"] and planned.get("expect_error"):
        flag = "expected error"
    print(f"{i:2d} {step['tool']:32s} {flag:15s} {step.get('seconds', 0):8.1f}s  {str(r.get('error', ''))[:100]}")
EOF
```

A good run has `tools_listed` 33, every step `ok` except step 15, the results the notes describe, and FoldX values in the three validations that use it. A step that is `ok` does not show that FoldX ran (section 4), so check those after the run, in the shell of section 3 (the lines use `$T` and `$RUN`); each line prints what it found:

```bash
W="$T/work"
# step 36: the FoldX row of the AUC summary, else the FoldX values per TF in the raw scores
grep ',FoldX,' "$W/method_comparison/auc_summary.csv" || python -c "import pandas as pd, sys; d = pd.read_csv(sys.argv[1]); print(d.groupby('tf')['FoldX'].count().to_dict() if 'FoldX' in d else 'no FoldX column: FoldX did not run')" "$W/method_comparison/raw_scores.csv"
# step 37: the FoldX row of the summary, and the pipeline's "FoldX OK" line, from the server's call log
grep '^FoldX,' "$W/remap_validation/summary.csv"; grep -o 'FoldX OK: pos [0-9]*/[0-9]*, neg [0-9]*/[0-9]*' "$T/calls_$RUN.jsonl"
# step 38: the variants with a FoldX ddG
python -c "import pandas as pd, sys; d = pd.read_csv(sys.argv[1]); print(d['foldx_ddG'].notna().sum(), 'of', len(d), 'variants have a FoldX ddG')" "$W/clinvar_validation/combined_results.csv"
```

- Step 36 is good with a line `NKX2-1,GATA1,FoldX,<auc>,20,20`. That row needs at least 10 FoldX scores per TF; without it, the second command prints how many windows per TF FoldX scored, and `no FoldX column` means that FoldX did not run at all.
- Step 37 is good with `FoldX,<auc>,20,20` and `FoldX OK: pos 20/20, neg 20/20`. A smaller count is the number of FoldX runs that succeeded; the FoldX row needs at least 5 on each side.
- Step 38 is good with a count above 0, ideally every variant (a variant keeps NaN when its window cannot be cut or a FoldX run fails). `0 of N` means that FoldX did not run or failed on every variant.

If a check fails, look at steps 33 and 34, which call FoldX directly and return its errors, and check that `foldx` is on the PATH the server gets (section 3). These checks were tried on the outputs of the three pipelines run in-process on the demo data, once with FoldX replaced by random numbers and once without FoldX (the method comparison and the ClinVar check, before version 1.0.0 made both stop without FoldX): they printed the FoldX rows and counts in the first case, and `no FoldX column` and `0 of 12` in the second.

For the validations, also look at the files under `$T/work/<step>/` (for example `remap_validation/summary.csv`, `clinvar_validation/summary_all_methods.csv`); for the SLURM jobs, `squeue -u $USER` after the run shows what is still queued.

Step 15 only shows that the sweep was created and its agent job submitted. Whether the trial trained shows in the agent job, which runs in the project directory and writes `sweep_<sweep id>_<job id>.out` and `.err` there; the job fails when a trial fails, and a trained trial leaves `best-<model>-<run name>.ckpt` under `saved_models_tuned/`:

```bash
read JOB SWEEP < <(python -c "import json, sys; r = [s for s in json.load(open(sys.argv[1]))['steps'] if s.get('tool') == 'tfbs_sweep'][0]['result']; print(r.get('job_id'), r.get('sweep_id'))" "$T/session_log_$RUN.json")
sacct -j "$JOB" --format=JobID,State,Elapsed,ExitCode
tail -n 25 "$T/project/sweep_${SWEEP}_${JOB}.err"
find "$T/project/saved_models_tuned" -name '*.ckpt'
```

## 7. Runtime

Measured on 2026-10-14 on node s-sc-gpu002 of the BIH cluster (one NVIDIA A100 40 GB, 25 CPUs, allocation 11194026) with tfbs-mcp 040496a and the INTERSECTIONS directory linked unmerged (3.69 million weighted lines, about 132 s of merging by the formula of 2.3); every one of the 49 steps succeeded and FoldX ran in all three validations that use it. WandB was logged in, so step 15 created a one-trial sweep; its trial failed on the placeholders of the sweep YAML, which wandb locks (the agent read `data_path` and `checkpoint_dir` as `/path/`), and the agent job still ended COMPLETED. Both were fixed afterwards: the agent now reads the task's values over the placeholders and exits nonzero when a trial fails. A first run on 2026-10-13 at 6fbec08 (node s-sc-gpu011) took the same time to within a minute per step, apart from `tfbs_seqlets` (45.6 s there).

| Steps | Measured time |
|---|---|
| server start | 29.9 s |
| 0 to 9, data prep (34,700 windows and negatives) | 17.6 s (the GC-matched extraction 15.7 s) |
| 10 to 15, training (6 epochs of 44,416 windows) and submissions | 57.2 s (local training 50.0 s, the sweep 6.8 s) |
| 16 to 31, analysis | 6.8 s (seqlets 3.5 s, every other call under 2.1 s) |
| 32 to 34, structure | 269.8 s (one BuildModel window 268.5 s, AnalyseComplex 0.8 s) |
| 35, benchmark (4 negative TFs, 2,000 windows each) | 137.3 s |
| 36, method comparison (two FoldX batches of 20) | 686.6 s |
| 37, ReMap (two FoldX batches of 20) | 674.8 s |
| 38, ClinVar (44 variants in the 1,000 sampled windows, two FoldX runs each) | 1074.6 s |
| 39 to 48, SLURM | under 0.1 s each |
| total | 2955 s from launch to the last step, about 49 minutes |

The estimate made before that run (25 to 40 minutes) was too low for the three FoldX validations, which took 11 to 18 minutes each; a node with fewer than 24 cores stretches them further.

## 8. The local dry run

The plan was checked on a Windows workstation without SLURM, FoldX, the `fimo` program, pysam or WandB, on CPU, against a non-editable install (`uv pip install` of a `git archive` of the branch into a fresh Python 3.13 venv) with the bundled demo files standing in for the cluster data: `demo_peaks.bed` as `nkx2-1_chipatlas.bed`, `demo_genome.fa`, `MA1994.1.meme`, a synthetic two-TF `chipatlas_intersect/`, PDB 9U19 as `@PDB`, empty files as `@CLINVAR`, and the variant written as `demo_chr8:830:T>G`. All 31 steps that can run there succeeded, among them the training, the 16 analysis calls, the RCSB download and the benchmark through the `models_dir` lookup. The other 18 failed only for the missing programs: `sbatch binary not found on PATH` (13, 14, 15, 43, 45), `FoldX binary not found on PATH.` (33, 34), the missing `fimo` program (36, 37, 38), and `$ref` errors for the `job_id` of the failed submissions (39 to 42, 44, 46 to 48). Step 34 stops at the missing FoldX before it reads the region; its one window was checked separately by calling the tool in-process with FoldX and pyBigWig replaced by stubs, on the demo genome with its sequence renamed to `chr8` (the tool accepts only chromosome names that start with `chr`): `chr8:824-848`, the demo position of the plan's region, gives `num_windows` 1 and scores `CTCACTTGACC`.
