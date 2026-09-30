# tfbs-mcp

A Python library and Model Context Protocol (MCP) server for transcription factor binding site analysis. The library packages neural network architectures, ChIP-seq data pipelines, attribution and motif analysis, and structure-based binding-energy scoring. The MCP server exposes 33 of these operations as tools that Claude can call directly, so Claude can compose a full TFBS analysis from a natural-language goal.

## Installation

Setup takes about 20 minutes, most of it waiting for PyTorch to download.

### What you need first

1. **Claude Code or Claude Desktop.** The server adds tools to Claude, so you need one of these two apps:
   - **Claude Code**, which runs in a terminal. This is the simpler setup (one command in step 3). It needs a paid Claude plan (Pro, Max, Team or Enterprise) or an Anthropic Console account. Install it by following <https://code.claude.com/docs/en/setup>.
   - **Claude Desktop**, the Claude app with a window. Download it from <https://claude.ai/download>.

   Other AI apps work too, but are not covered here. See [docs/other-clients.md](docs/other-clients.md).
2. **Python 3.10 to 3.13.** Python 3.14 does not work yet: one dependency (pybigtools) has no 3.14 build and fails to compile. Check your version with `python --version` (on macOS and Linux, `python3 --version`). If you need to install Python, take 3.13 from <https://www.python.org/downloads/> and, on Windows, tick **"Add python.exe to PATH"** in the installer.
3. **About 3 GB of free disk space**, mostly for PyTorch.

You never start the server by hand. Once it is connected in step 3, Claude starts it in the background whenever it needs a tool.

### Step 1: Create an environment

An environment is a private folder for this project's Python packages, so the install cannot break anything else on your computer. A conda or micromamba environment with Python 3.13 works just as well; if you use one of those, create and activate it, then go to step 2.

Open a terminal (Windows: press the Windows key, type `powershell`, open **Windows PowerShell**; macOS: Cmd+Space, type `terminal`, press Enter) and run:

Windows PowerShell:

```powershell
cd $HOME
python -m venv tfbs-env
tfbs-env\Scripts\Activate.ps1
```

macOS / Linux:

```bash
cd ~
python3 -m venv tfbs-env
source tfbs-env/bin/activate
```

Your prompt now starts with `(tfbs-env)`, which means the environment is active. In a new terminal window you have to run the last line again before using `pip` or `tfbs-mcp`.

> If PowerShell says `Activate.ps1 cannot be loaded because running scripts is disabled`, run the following once, answer `Y`, and activate again:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

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
  [ok  ] executable: C:\Users\you\tfbs-env\Scripts\tfbs-mcp.exe

Everything needed is installed. Connect the server with ONE of these:
...
```

If a line says `FAIL`, it also says how to fix it. The command then prints the exact setup text for step 3, with the path to your installation already filled in.

### Step 3: Connect Claude

Use the section for the app you have.

**Claude Code.** With the environment still active, run the command for your system:

Windows PowerShell:

```powershell
claude mcp add tfbs --scope user -- (Get-Command tfbs-mcp).Source
```

macOS / Linux:

```bash
claude mcp add tfbs --scope user -- "$(which tfbs-mcp)"
```

(This is the same `claude mcp add` line that `tfbs-mcp --check` printed.) Then run `claude mcp list`: `tfbs` should be listed as connected. Inside a Claude Code session, `/mcp` shows the same.

**Claude Desktop.** Open **Settings > Developer > Edit Config**. This opens `claude_desktop_config.json`, and creates it if it does not exist yet. Paste the block that `tfbs-mcp --check` printed; it looks like this:

```json
{
  "mcpServers": {
    "tfbs": {
      "command": "C:\\Users\\you\\tfbs-env\\Scripts\\tfbs-mcp.exe",
      "args": []
    }
  }
}
```

Use the block from `--check` rather than this example: it contains your real path, with the double backslashes that Windows paths need in this file. If the file already contains `"mcpServers"`, add only the `"tfbs": {...}` entry inside it, with a comma after the previous entry. Save, then quit Claude Desktop completely (closing the window is not enough) and open it again.

### Step 4: Test it

Ask Claude:

> Check the TFBS server environment. Is it running on SLURM, does it see a GPU, and what models are loaded?

Claude calls the tools `tfbs_server_info` and `tfbs_list_models`, which need no data files, and reports the device, the GPU count and an empty model list.

- The first tool call can take up to a minute, because the server loads PyTorch then. Later calls are fast.
- If something goes wrong in a tool, the error comes back as a normal answer such as `{"error": "...", "fix": "..."}`, not as a crash. If an answer looks wrong, ask Claude to show the raw tool response.

### Troubleshooting

In most cases, run `tfbs-mcp --check` in the activated environment first. It names the problem.

| What you see | Why | Fix |
|---|---|---|
| `No module named 'mcp.server.fastmcp'`, or `tfbs-mcp: mcp 2.x is installed` | mcp 2 renamed the API this server uses | `pip install "mcp>=1.28,<2"` in the same environment |
| `tfbs` shows a red X or `failed` in `/mcp`, or reconnecting gives `-32000` | The server crashed at startup | Run `tfbs-mcp --check`; fix the `FAIL` line |
| pip fails on `pybigtools` with `Failed to build a native library through cargo` | Python 3.14 | Recreate the environment with Python 3.13 |
| `No matching distribution found for torch` | An old Linux system (glibc older than 2.28, e.g. CentOS 7 or RHEL 7), where current PyTorch does not install | Use Python 3.11 and update pip (`pip install -U pip`); check glibc with `ldd --version` |
| `The term 'tfbs-mcp' is not recognized` / `command not found` | The environment is not active | Run the activation line from step 1 |
| `tfbs` missing from Claude Desktop, no error | The config file is not valid JSON, usually a single backslash or a trailing comma | Paste the block from `tfbs-mcp --check` again |
| Works in the terminal, not in the app | The config uses the bare name `tfbs-mcp` instead of the full path | Use the path that `tfbs-mcp --check` prints |
| `{"error": "No reference genome FASTA configured."}` | The tool needs data you have not supplied | See [What needs extra setup](#what-needs-extra-setup) |
| Claude Code times out while the server starts | Slow disk, or antivirus scanning the environment | Set `MCP_TIMEOUT=60000` before starting `claude` |

## What needs extra setup

Five of the 33 tools work with nothing but the install: `tfbs_server_info`, `tfbs_list_models`, `tfbs_preprocess`, `tfbs_classify_metrics`, and `tfbs_fetch_structure` (which needs internet). `tfbs_config` is not one of them: it errors unless the `data_path` you give it already exists. The rest need data or software you supply:

| To use | You need | Point it there with |
|---|---|---|
| Sequence extraction, benchmarks | A reference genome FASTA (hg38 is ~3 GB) | `TFBS_GENOME_FASTA` |
| `tfbs_fimo`, motif scanning | The `fimo` binary from the [MEME Suite](https://meme-suite.org/) on your PATH, plus a JASPAR/MEME motif file | `TFBS_MOTIF_FILE` |
| Structure-based scoring | **FoldX**, a separately licensed binary from <https://foldxsuite.crg.eu/>, plus a repaired TF-DNA PDB. The `foldx` pip extra installs pyBigWig, not FoldX | `TFBS_FOLDX_PDB` |
| Disease-variant validation | An indexed ClinVar VCF (`.vcf.gz` + `.tbi`) | `TFBS_CLINVAR_VCF` |
| Prediction, attribution | A trained model checkpoint; none ship with the package | `tfbs_load_model` |
| `slurm_*`, and any `slurm=true` option | An HPC cluster with SLURM | (none) |
| Training, benchmark, validation tools | A git clone (they run scripts under `scripts/`, which pip does not install) | `TFBS_PROJECT_ROOT`, else the directory containing the installed `tfbs` package |

Set the environment variables so the tools find your data without you repeating the path every time.

Claude Code:

```bash
claude mcp add tfbs --scope user -e TFBS_GENOME_FASTA=/data/hg38/hg38.fa -e TFBS_MOTIF_FILE=/data/JASPAR/MA1994.1.meme -- "$(which tfbs-mcp)"
```

Claude Desktop, in the `tfbs` entry of the config file:

```json
{
  "mcpServers": {
    "tfbs": {
      "command": "C:\\Users\\you\\tfbs-env\\Scripts\\tfbs-mcp.exe",
      "args": [],
      "env": {
        "TFBS_GENOME_FASTA": "D:\\data\\genomes\\hg38\\hg38.fa",
        "TFBS_MOTIF_FILE": "D:\\data\\JASPAR\\MA1994.1.meme"
      }
    }
  }
}
```

Any tool argument you pass explicitly overrides these. If neither is set, the tool tells you which argument and which variable would fix it.

The repo-only training and benchmark scripts under `scripts/` use one further variable, `TFBS_PROJECT_ROOT`, for the directory holding `data/`, `saved_models/` and `results/`. The MCP tools that call those scripts read it too, falling back to the directory containing the installed `tfbs` package. It defaults to the repo root, so a fresh clone runs unedited; set it if your data or your checkout lives elsewhere:

```bash
export TFBS_PROJECT_ROOT=/mnt/lab/tfbs
```

The example YAML configs under `scripts/` contain `/path/to/...` placeholders rather than real paths. Copy one and fill in your own before using it.

## Optional extras

The install command in step 2 already includes everything the server needs. This table is only for choosing a different set:

| Extra | Installs | Needed for |
|---|---|---|
| `mcp` | the MCP protocol library | the server itself |
| `genomics` | pyfaidx, tangermeme | reading a genome FASTA |
| `viz` | captum, tangermeme, logomaker, seaborn | attribution, mutagenesis, seqlets |
| `tracking` | wandb | experiment logging during training |
| `foldx` | pyBigWig | not FoldX itself, see [What needs extra setup](#what-needs-extra-setup) |
| `equinet` | equirc | the `EquiNet` architecture only |
| `dev` | pytest | running the test suite (also needs `mcp` and `viz`) |
| `all` | all of the above | |

## For developers

### Library only

To use the Python library without the MCP server:

```bash
pip install git+https://github.com/sproft/tfbs-mcp.git
```

PyTorch is installed automatically. If you need a specific CUDA build, install torch first following <https://pytorch.org/>, then this package.

### Library

```python
import tfbs.nn.models as models
from tfbs.dataloaders.SeqDataset import TFBSDataModule

model = models.VCNNBpnet.load_from_checkpoint("path/to/best.ckpt")
data = TFBSDataModule(data_path="data/CTCF", batch_size=64)
```

### Training

```bash
tfbs-train fit --config my_config.yaml
```

`tfbs-train` is installed by pip, but the example configs live in the repo and are not packaged. Clone the repository to get `scripts/cli/config.example.yaml`, copy it, and replace the `/path/to/...` placeholders with your own data and checkpoint locations. The `tfbs_config` MCP tool generates a fresh config if you would rather start from scratch.

### Prediction

```bash
tfbs-predict --peak_file peaks.bed --model_name my_model --config_file config.yml
```

### Running the server by hand

`tfbs-mcp` with no arguments starts the server and waits silently for an MCP client on standard input; press Ctrl+C to quit. To talk to it without Claude, use the MCP Inspector:

```bash
npx -y @modelcontextprotocol/inspector "$(which tfbs-mcp)"
```

## Available tools (MCP server)

The 33 tools are organized into six layers:

| Layer | Tools | Examples |
|---|---|---|
| Inference | 10 | `tfbs_predict`, `tfbs_analyze` (DeepLiftShap), `tfbs_mutagenesis`, `tfbs_fimo` |
| SLURM management | 5 | `slurm_submit`, `slurm_status`, `slurm_logs` |
| Training | 4 | `tfbs_config`, `tfbs_train`, `tfbs_sweep`, `tfbs_validate` |
| Data pipeline | 7 | `tfbs_extract_loci`, `tfbs_prepare_dataset`, `tfbs_seqlets`, `tfbs_epistasis` |
| Structure-based | 3 | `tfbs_fetch_structure`, `tfbs_binding_energy`, `tfbs_binding_scan` |
| Validation and comparison | 4 | `tfbs_chipseq_benchmark`, `tfbs_chipseq_method_comparison`, `tfbs_remap_validation`, `tfbs_disease_variant_validation` |

## Available model architectures

All models inherit from `BaseModel` (a PyTorch Lightning `LightningModule`) and support both classification and regression:

- **VCNNBpnet**: Dilated CNN with dense layers (primary architecture for short windows)
- **RNN**: GRU-based recurrent network
- **SimpleCNN**, **CNN**, **FlexibleCNN**, **VCNN**, **my_cnn**: Convolutional variants
- **MLP**, **FlexibleMLP**: Fully connected networks
- **BPNet**, **BPNetReal**: BPNet architectures for base-resolution count data
- **MultiCNN**: CNN that also consumes numerical covariates
- **EquiNet**: Reverse-complement equivariant network (needs the `equinet` extra)

Each has a ready-to-use config in `scripts/cli/yamls/models/`. Compose one with the full example config:

```bash
tfbs-train fit --config scripts/cli/config.example.yaml --model scripts/cli/yamls/models/VCNNBpnet.yaml
```

`VCNNBpnet.yaml` and `RNN.yaml` carry the tuned settings from `do_train_final.sh`; the rest use class defaults. All thirteen instantiate at a 24 bp window, and eleven accept one-hot `(batch, 4, L)` input directly. `FlexibleMLP` expects tabular features and `MultiCNN` also needs a covariate tensor, as noted in those files.

`EquiNet` additionally needs the optional `equirc` package. `tfbs-mcp` is not on PyPI, so install the dependency directly (it is on PyPI):

```bash
pip install equirc
```

Without it, constructing an `EquiNet` raises an `ImportError` naming that command. Every other architecture works without the extra.

## Project structure

```
tfbs/                     Python package
├── nn/models.py          Model architectures (VCNNBpnet, RNN, CNN, MLP, BPNet)
├── dataloaders/          Data loading and one-hot encoding
├── prediction/           Window-based scoring and analysis
├── predict.py            ChIP-seq peak prediction CLI
├── cli/train_cli.py      Lightning CLI backing the tfbs-train command
└── mcp/                  MCP server
    ├── cli.py            tfbs-mcp command: version guard and --check
    ├── server.py         FastMCP server, 30 of 33 tools
    └── foldx_tools.py    Structure-based tools

scripts/                  Repo-only analysis scripts (not installed by pip)
├── cli/                  Training configs and SLURM job templates
├── chipseq_benchmark*.py ChIP-seq classification benchmarks
├── disease_variant_validation.py
├── remap_validation.py   ReMap external validation
└── ...

docs/other-clients.md     Setup for AI apps other than Claude
tests/                    pytest suite
```

## Running tests

```bash
pip install ".[dev,mcp,viz,genomics]"
pytest
```

The `dev` extra is only pytest. The suite imports `tfbs.mcp.server`, so it needs `mcp` to collect at all, and the mutagenesis/marginalize tests exercise real tangermeme (from `viz` or `genomics`). Installing `dev` alone is not enough.

## Citation

If you use `tfbs-mcp` in your work, please cite:

> Proft S, Leser U. tfbs-mcp: a Model Context Protocol server for agentic transcription factor binding site analysis. *Bioinformatics*, 2026 (in preparation).

## License

MIT. See [LICENSE](LICENSE) for details.
