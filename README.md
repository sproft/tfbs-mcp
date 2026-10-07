# tfbs-mcp

A Python library and Model Context Protocol (MCP) server for transcription factor binding site analysis. The library packages neural network architectures, ChIP-seq data pipelines, attribution and motif analysis, and structure-based binding-energy scoring. The MCP server exposes 33 of these operations as tools that Claude can call directly, so Claude can compose a full TFBS analysis from a natural-language goal.

This page explains how to install the server and connect it to Claude. The Python library, model training, the test suite and the project layout are described in [docs/development.md](docs/development.md). AI apps other than Claude are covered in [docs/other-clients.md](docs/other-clients.md).

## Installation

Setup takes about 20 minutes, most of it waiting for PyTorch to download. The commands are written for Linux. macOS uses the same commands. Windows differs in two places, listed under [Windows](#windows) at the end of this section.

### What you need

1. **Claude Code or Claude Desktop.** The server adds tools to Claude, so you need one of the two:
   - **Claude Code** runs in a terminal. It needs a paid Claude plan (Pro, Max, Team or Enterprise) or an Anthropic Console account. Install it by following the Claude Code quickstart at <https://code.claude.com/docs/en/quickstart>.
   - **Claude Desktop** is the Claude app with a window. Download it from <https://claude.com/download>.

   Other AI apps work too, but are not covered here. See [docs/other-clients.md](docs/other-clients.md).
2. **Python 3.10 to 3.13**, provided by an environment manager. We recommend micromamba, which installs Python for you: follow its install guide at <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>. If a Python in that range is already installed, its built-in `venv` module works too: <https://docs.python.org/3/library/venv.html>.
3. **git**, because the install command fetches the package from GitHub. Install it with your package manager (for example `sudo apt install git`), or inside the environment with `micromamba install -c conda-forge git`.
4. **About 3 GB of free disk space**, mostly for PyTorch.
5. **A GPU is optional.** Everything runs on the CPU. A CUDA GPU is used automatically when one is present.

### Step 1: Create an environment

An environment is a private folder for this project's Python packages, so the install cannot break anything else on your computer.

With micromamba:

```bash
micromamba create -n tfbs-mcp python=3.13 -c conda-forge
micromamba activate tfbs-mcp
```

Your prompt now starts with `(tfbs-mcp)`, which means the environment is active. In a new terminal window, run the `micromamba activate` line again before using `pip` or `tfbs-mcp`.

If `micromamba activate` instead says the shell is not initialised, run the `micromamba shell init` command it suggests, open a new terminal, and run the activate line again. This happens once, right after installing micromamba.

With venv instead (Python 3.10 to 3.13 must already be installed):

```bash
python3 -m venv ~/tfbs-env
source ~/tfbs-env/bin/activate
```

Here the prompt starts with `(tfbs-env)`, and the `source` line is the one to repeat in a new terminal.

### Step 2: Install and check

```bash
pip install "tfbs-mcp[mcp,genomics,viz] @ git+https://github.com/sproft/tfbs-mcp.git"
```

This downloads PyTorch, roughly 2 to 3 GB, and can take 5 to 20 minutes. It is not stuck.

Then check the installation:

```bash
tfbs-mcp --check
```

A working install ends like this (the paths will be your own):

```
tfbs-mcp installation check

  [ok  ] Python 3.13.12
  [ok  ] mcp 1.30.0
  [ok  ] server loads, 33 tools registered
  [ok  ] PyTorch 2.14.0
  [ok  ] tangermeme 1.5.0
  [ok  ] executable: /home/you/micromamba/envs/tfbs-mcp/bin/tfbs-mcp

Everything needed is installed. Connect the server with ONE of these:
...
```

If a line says `FAIL`, it also says how to fix it. The command then prints the exact setup text for step 3, with the path to your installation already filled in.

### Step 3: Connect Claude

Use the part for the app you have.

**Claude Code.** With the environment still active, run:

```bash
claude mcp add tfbs --scope user -- "$(which tfbs-mcp)"
```

This does the same as the `claude mcp add` line that `tfbs-mcp --check` printed; either one works. Then list the servers:

```bash
claude mcp list
```

The list should contain `tfbs` with a connected status. Inside a Claude Code session, `/mcp` shows the same.

**Claude Desktop.** Open **Settings > Developer > Edit Config**. This opens `claude_desktop_config.json`, and creates it if it does not exist yet. Paste the block that `tfbs-mcp --check` printed; it looks like this:

```json
{
  "mcpServers": {
    "tfbs": {
      "command": "/home/you/micromamba/envs/tfbs-mcp/bin/tfbs-mcp",
      "args": []
    }
  }
}
```

Use the block from `--check` rather than this example, because it contains your real path. If the file already contains `"mcpServers"`, add only the `"tfbs": {...}` entry inside it, with a comma after the previous entry. Save, then quit Claude Desktop completely (closing the window is not enough) and open it again.

You never start the server by hand. From now on, Claude starts it in the background whenever it needs a tool.

### Step 4: Test it

Start Claude. For Claude Code, run this in the terminal:

```bash
claude
```

For Claude Desktop, open the app.

Now type the following line into Claude's chat, the box where you normally write to Claude. It is a message for Claude, not a shell command. If you paste it into the terminal instead, the shell answers with something like `Command 'Check' not found`.

> Check the TFBS server environment. Is it running on SLURM, does it see a GPU, and what models are loaded?

Claude calls the tools `tfbs_server_info` and `tfbs_list_models`, which need no data files, and summarises what they return. The reply should contain these points:

- **SLURM: No.** Correct unless you are running on an HPC cluster with the SLURM scheduler.
- **GPU count and default device.** For example 1 GPU and device `cuda` with batch size 512, or 0 GPUs and device `cpu` with batch size 64. Both are fine: the tools work on the CPU.
- **Models loaded: None.** Expected. No model is loaded until you load one with `tfbs_load_model`.
- **A note that PyTorch has not been imported yet.** Normal. The server loads PyTorch on the first call that needs it, which is why that call can take up to a minute. Later calls are fast.

One detail: before PyTorch is loaded, the device is detected from the environment (SLURM variables, `CUDA_VISIBLE_DEVICES` or `nvidia-smi`). On a computer with an NVIDIA GPU but a CPU-only PyTorch build, the first answer says `cuda` and a later one says `cpu`. That is the detection catching up, not a fault.

If something goes wrong in a tool, the error comes back as a normal answer such as `{"error": "...", "fix": "..."}`, not as a crash. If an answer looks wrong, ask Claude to show the raw tool response.

### Step 5: Use it

Everything in this step is typed into Claude's chat, like the test prompt above.

**Without any data.** These prompts work with nothing but the install:

> One-hot encode these two 16 bp sequences and add their reverse complements: AAACACTTGAGGCAAA and TTTGCCTCAAGTGTTT.

Claude calls `tfbs_preprocess` and reports a tensor of shape `[4, 4, 16]` for 4 sequences: the two inputs plus their reverse complements. The first sequence contains `CACTTGA`, the core of the NKX2-1 motif, and the second is its reverse complement.

> Compute ROC-AUC and PR-AUC for a classifier that scored the positive examples 0.9, 0.8, 0.7, 0.6 and the negative examples 0.3, 0.2, 0.4, 0.1.

Claude calls `tfbs_classify_metrics` and reports ROC-AUC 1.0 and PR-AUC 1.0, because every positive scores above every negative.

> Download PDB entry 3RKQ, an NKX2-5 homeodomain bound to DNA, from RCSB and tell me where the file was saved.

Claude calls `tfbs_fetch_structure` with the PDB id `3RKQ`, downloads the structure from RCSB into a temporary folder and returns the path. This is the one prompt here that needs internet access, and it writes a file.

> Check the TFBS server environment again. Now that PyTorch is loaded, which device and batch size will it use?

Claude calls `tfbs_server_info` once more. The first two prompts made the server load PyTorch, so the answer now also includes the PyTorch version and whether CUDA is available.

Setup is complete. The remaining tools need reference files such as a genome, which you can add at any time; see [Adding your data](#adding-your-data), which starts with a bundled example that needs no download. The full list of tools is under [Available tools](#available-tools).

### Windows

The steps above work in Windows PowerShell with two differences. The micromamba install guide linked above covers Windows as well; once micromamba is installed, the `micromamba` lines in step 1 are the same.

Creating and activating a venv (step 1) uses `python` instead of `python3` and a different activation line:

```powershell
cd $HOME
python -m venv tfbs-env
tfbs-env\Scripts\Activate.ps1
```

If PowerShell says `Activate.ps1 cannot be loaded because running scripts is disabled`, run the following once, answer `Y`, and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Connecting Claude Code (step 3) uses the PowerShell way of finding the executable:

```powershell
claude mcp add tfbs --scope user -- (Get-Command tfbs-mcp).Source
```

In `claude_desktop_config.json`, every backslash in a Windows path is written twice, as in `C:\\Users\\you\\...`. The blocks printed by `tfbs-mcp --check` and `tfbs-mcp --setup` already have them, so paste those rather than typing a path.

### Troubleshooting

In most cases, run `tfbs-mcp --check` in the activated environment first. It names the problem.

| What you see | Why | Fix |
|---|---|---|
| `Command 'Check' not found`, `Check: command not found` or `The term 'Check' is not recognized` | The test prompt from step 4 was typed into the terminal instead of Claude's chat | Start Claude (`claude` in the terminal, or open Claude Desktop) and type the prompt into the chat |
| `No module named 'mcp.server.fastmcp'`, or `tfbs-mcp: mcp 2.x is installed` | mcp 2 renamed the API this server uses | `pip install "mcp>=1.28,<2"` in the same environment |
| `tfbs` shows a red X or `failed` in `/mcp`, or reconnecting gives `-32000` | The server crashed at startup | Run `tfbs-mcp --check`; fix the `FAIL` line |
| pip fails on `pybigtools` with `Failed to build a native library through cargo` | Python 3.14, which one dependency (pybigtools) cannot be built for yet | Recreate the environment with Python 3.13 |
| `No matching distribution found for torch` | An old Linux system (glibc older than 2.28, e.g. CentOS 7 or RHEL 7), where current PyTorch does not install | Use Python 3.11 and update pip (`pip install -U pip`); check glibc with `ldd --version` |
| `tfbs-mcp: command not found` / `The term 'tfbs-mcp' is not recognized` | The environment is not active | Run the activation line from step 1 |
| `Cannot find command 'git'` during `pip install` | pip needs git to fetch the package from GitHub | Install git (see [What you need](#what-you-need)) and run the install command again |
| `tfbs` missing from Claude Desktop, no error | The config file is not valid JSON, usually a single backslash or a trailing comma | Paste the block from `tfbs-mcp --check` again |
| Works in the terminal, not in the app | The config uses the bare name `tfbs-mcp` instead of the full path | Use the path that `tfbs-mcp --check` prints |
| `{"error": "No reference genome FASTA configured."}` | The tool needs data you have not supplied | See [Adding your data](#adding-your-data) |
| `Compressed FASTA is only supported in BGZF format` | The genome is a plain gzip file; `hg38.fa.gz` from UCSC is one | `gunzip hg38.fa.gz` and register the uncompressed `hg38.fa` |
| `tfbs_fimo` fails with `FileNotFoundError` | The `fimo` program is not on your PATH | Install the MEME Suite (see [Adding your data](#adding-your-data)) and restart Claude |
| Claude Code times out while the server starts | Slow disk, or antivirus scanning the environment | Set `MCP_TIMEOUT=60000` before starting `claude` |

## Adding your data

You do not need any of this to finish setup: steps 1 to 5 above are complete without it. Five tools work with nothing but the install (`tfbs_server_info`, `tfbs_list_models`, `tfbs_preprocess`, `tfbs_classify_metrics`, and `tfbs_fetch_structure`, which needs internet). Each tool group below needs one input. Get it from the link in the table, then run `tfbs-mcp --setup` once: it prints the Claude Code command and the Claude Desktop block with your paths filled in, and can run the Claude Code command for you.

### Try it first with the bundled example

The package ships a 10 kb slice of human chromosome 8 around the thyroglobulin promoter (thyroglobulin is the classic NKX2-1 target; the study's EMSA-seq libraries were anchored on the NKX2-1 site of its rat promoter), a BED file with four NKX2-1 motif hits in that slice, and the JASPAR `MA1994.1` motif. Register them with:

```bash
tfbs-mcp --setup --demo
```

It prints the path of the demo BED file. Then, in Claude's chat:

> Extract 200 bp sequences around every peak in <the demo_peaks.bed path it printed> and save the tensors to /tmp/tfbs-demo.

Claude calls `tfbs_extract_loci` on the bundled slice and reports four 200 bp sequences, one-hot encoded as a tensor of shape `[4, 4, 200]`. With the MEME Suite installed:

> Scan the sequence CTCACTTGACCTT for the NKX2-1 motif with FIMO.

That 13-mer is the strongest match in the slice, about 4.2 kb upstream of the thyroglobulin transcription start, so `tfbs_fimo` reports a hit. Where the files come from and how they were cut is documented in `tfbs/data/demo/README.md` in the repository. When you have your own data, run `tfbs-mcp --setup` again without `--demo`; it replaces the registration.

### Your own data

| What it enables | What you need | Where to get it | Variable |
|---|---|---|---|
| Sequence extraction and the ChIP-seq benchmarks (`tfbs_extract_loci`, `tfbs_chipseq_benchmark`, ...) | A reference genome FASTA, uncompressed (hg38 is about 3 GB uncompressed) | hg38 from UCSC: <https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz> (just under 1 GB), then `gunzip hg38.fa.gz`. Register the resulting `hg38.fa`; the server writes an index next to it on first use. NCBI Datasets offers the same assembly (GRCh38.p14, <https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.40/>); note that NCBI names sequences by accession (`NC_000008.11`, not `chr8`), and the chromosome names in your BED files must match the record names in whichever FASTA you register | `TFBS_GENOME_FASTA` |
| Peaks to extract from or train on | A BED file of ChIP-seq peaks for your transcription factor | ChIP-Atlas: <https://chip-atlas.org/> | none; name the file in your prompt |
| Motif scanning (`tfbs_fimo`) | The `fimo` program on your PATH, plus a MEME-format motif file | MEME Suite: <https://meme-suite.org/meme/doc/download.html>. Motif: JASPAR `MA1994.1` (NKX2-1) from <https://jaspar.elixir.no/api/v1/matrix/MA1994.1.meme>, licensed CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>) | `TFBS_MOTIF_FILE` |
| Structure-based scoring (`tfbs_binding_energy`, `tfbs_binding_scan`) | **FoldX**, a separately licensed program (free for academic use after registration; the `foldx` pip extra installs pyBigWig, not FoldX), plus a repaired TF-DNA complex PDB | FoldX from the FoldX Suite site: <https://foldxsuite.crg.eu/>. PDB: `tfbs_fetch_structure` downloads one from RCSB, for example <https://www.rcsb.org/structure/3RKQ>; `tfbs_binding_energy` can run FoldX `RepairPDB` on it with its `repair` option | `TFBS_FOLDX_PDB` |
| Disease-variant validation (`tfbs_disease_variant_validation`) | An indexed ClinVar VCF: `clinvar.vcf.gz` and its `.tbi` index side by side | NCBI, GRCh38: <https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz> (about 200 MB) and <https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi>. Both are refreshed regularly, so sizes change | `TFBS_CLINVAR_VCF` |
| Prediction and attribution (`tfbs_predict`, `tfbs_analyze`, `tfbs_mutagenesis`, ...) | A trained model checkpoint; none ship with the package | Train one on peaks from ChIP-Atlas (<https://chip-atlas.org/>) following [Training](docs/development.md#training) | none; load it with `tfbs_load_model` |
| `slurm_*` tools, and any `slurm=true` option | An HPC cluster running the SLURM scheduler | Your cluster's documentation; the scheduler itself is at <https://slurm.schedmd.com/> | none |
| Training, benchmark and validation tools (`tfbs_train`, `tfbs_sweep`, `tfbs_validate`, `tfbs_chipseq_benchmark`, ...) | A git clone of this repository, because these tools run scripts under `scripts/`, which pip does not install | `git clone https://github.com/sproft/tfbs-mcp.git` (<https://github.com/sproft/tfbs-mcp>) | `TFBS_PROJECT_ROOT` |

`TFBS_PROJECT_ROOT` is the directory holding `data/`, `saved_models/` and `results/`. The repo-only scripts under `scripts/` read it too. It defaults to the directory containing the installed `tfbs` package, which is the repository root in a clone, so a fresh clone runs unedited; set it if your data or your checkout lives elsewhere. The example configs and the `tfbs_config` tool are described under [Training](docs/development.md#training).

Any tool argument you pass explicitly overrides these variables. If neither is set, the tool tells you which argument and which variable would fix it.

### Registering the paths with tfbs-mcp --setup

Run this once in the activated environment, after downloading what you need:

```bash
tfbs-mcp --setup
```

What it does:

- It asks for each of the five variables in turn, one question per variable. Press Enter to skip one.

  | Variable | What it asks for |
  |---|---|
  | `TFBS_GENOME_FASTA` | reference genome FASTA (an existing file) |
  | `TFBS_MOTIF_FILE` | MEME-format motif file (an existing file) |
  | `TFBS_CLINVAR_VCF` | indexed ClinVar VCF, the `.vcf.gz` file (an existing file; a missing `.tbi` next to it gives a warning, not an error) |
  | `TFBS_FOLDX_PDB` | repaired TF-DNA PDB (an existing file) |
  | `TFBS_PROJECT_ROOT` | directory holding `data/`, `saved_models/` and `results/` (an existing directory) |

- A path that does not exist is reported and the question is asked again; Enter skips it.
- It then prints both setup forms with the chosen variables filled in and the skipped ones left out: the Claude Code command, `claude mcp add tfbs --scope user --env KEY=VALUE [KEY2=VALUE2 ...] -- "<absolute path of tfbs-mcp>"`, and the Claude Desktop JSON block, whose `tfbs` entry gains an `"env"` object listing the variables (no `"env"` when nothing was set).
- If the `claude` command is on your PATH, it asks `Register with Claude Code now? [y/N]`. Answer `y` and it runs `claude mcp remove tfbs --scope user` (a failure because `tfbs` was not registered yet is ignored), then the `claude mcp add` command, and reports success or the error. Answer `N`, or run it on a computer without `claude`, and it only prints.
- It never edits `claude_desktop_config.json`. For Claude Desktop, open **Settings > Developer > Edit Config**, replace the earlier `tfbs` entry with the printed block, save, then quit and reopen the app.

Flags prefill answers, so you are not asked for those variables: `--genome PATH`, `--motif PATH`, `--clinvar PATH`, `--foldx-pdb PATH`, `--project-root PATH`. A prefilled path is still checked. `--non-interactive` asks nothing at all: it uses only the flags, never offers to register, and just prints both forms. For example:

```bash
tfbs-mcp --setup --genome /data/hg38/hg38.fa --motif /data/JASPAR/MA1994.1.meme --non-interactive
```

Run `--setup` again whenever a path changes. The `remove` step replaces the earlier registration, so nothing accumulates. Restart Claude Code afterwards; Claude Desktop needs the edited config and a full quit and reopen.

Where the variables end up: for Claude Code, in the user-scope `tfbs` entry of `~/.claude.json` (`claude mcp get tfbs` shows it; the file also holds your sign-in, so do not edit it by hand). For Claude Desktop, in the `env` block of the `tfbs` entry in `claude_desktop_config.json`. Claude passes them to the server's environment every time it starts it.

### Setting the variables by hand

If you would rather not use `--setup`, this is what it writes.

Claude Code (the `remove` comes first because adding a name that already exists at the same scope is an error):

```bash
claude mcp remove tfbs --scope user
claude mcp add tfbs --scope user --env TFBS_GENOME_FASTA=/data/hg38/hg38.fa TFBS_MOTIF_FILE=/data/JASPAR/MA1994.1.meme -- "$(which tfbs-mcp)"
```

Claude Desktop, in the `tfbs` entry of the config file:

```json
{
  "mcpServers": {
    "tfbs": {
      "command": "/home/you/micromamba/envs/tfbs-mcp/bin/tfbs-mcp",
      "args": [],
      "env": {
        "TFBS_GENOME_FASTA": "/data/hg38/hg38.fa",
        "TFBS_MOTIF_FILE": "/data/JASPAR/MA1994.1.meme"
      }
    }
  }
}
```

### Example prompts once your data is registered

With a genome FASTA registered and a BED file of peaks on disk:

> Extract 200 bp sequences centred on every peak in /data/peaks/nkx2-1.bed and save the tensors to /data/tensors/nkx2-1.

Claude calls `tfbs_extract_loci`, reads the genome from `TFBS_GENOME_FASTA`, writes `sequences.pt` (and `scores.pt` when the BED file has a score column) into the output folder, and reports the tensor shape and how many peaks were extracted.

With the MEME Suite installed and the JASPAR motif `MA1994.1` (NKX2-1) registered as the motif file:

> Scan the sequence AAACACTTGAGGCAAA for the NKX2-1 motif with FIMO.

Claude calls `tfbs_fimo`, which runs the `fimo` program with `TFBS_MOTIF_FILE` against the sequence and returns the matches with position, strand, score and p-value.

## Uninstall

Remove the server from Claude. For Claude Code:

```bash
claude mcp remove tfbs --scope user
```

For Claude Desktop, open **Settings > Developer > Edit Config**, delete the `"tfbs": {...}` entry from `"mcpServers"` (and the comma that separated it from a neighbouring entry), save, then quit and reopen the app.

Remove the package:

```bash
pip uninstall tfbs-mcp
```

Or remove the whole environment instead. With micromamba, deactivate it first, because an active environment cannot be removed:

```bash
micromamba deactivate
micromamba env remove -n tfbs-mcp
```

With venv, delete the folder:

```bash
rm -rf ~/tfbs-env
```

Finally, delete the data files you downloaded for [Adding your data](#adding-your-data) (the genome, motif, ClinVar and PDB files) if you no longer need them. Structures fetched by `tfbs_fetch_structure` without an output folder sit in temporary folders named `tfbs_struct_*`.

## Available tools

The 33 tools are organized into six layers:

| Layer | Tools | Examples |
|---|---|---|
| Inference | 10 | `tfbs_predict`, `tfbs_analyze` (DeepLiftShap), `tfbs_mutagenesis`, `tfbs_fimo` |
| SLURM management | 5 | `slurm_submit`, `slurm_status`, `slurm_logs` |
| Training | 4 | `tfbs_config`, `tfbs_train`, `tfbs_sweep`, `tfbs_validate` |
| Data pipeline | 7 | `tfbs_extract_loci`, `tfbs_prepare_dataset`, `tfbs_seqlets`, `tfbs_epistasis` |
| Structure-based | 3 | `tfbs_fetch_structure`, `tfbs_binding_energy`, `tfbs_binding_scan` |
| Validation and comparison | 4 | `tfbs_chipseq_benchmark`, `tfbs_chipseq_method_comparison`, `tfbs_remap_validation`, `tfbs_disease_variant_validation` |

## Further documentation

- [docs/development.md](docs/development.md): the Python library, training and prediction commands, running the server by hand, optional extras, model architectures, project structure, and the test suite.
- [docs/other-clients.md](docs/other-clients.md): connecting AI apps other than Claude.

## Citation

If you use `tfbs-mcp` in your work, please cite:

> Proft S, Leser U. tfbs-mcp: a Model Context Protocol server for agentic transcription factor binding site analysis. *Bioinformatics*, 2026 (in preparation).

## License

MIT. See [LICENSE](LICENSE) for details.
