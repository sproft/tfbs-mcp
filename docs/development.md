# Developer guide

This page is for people who want the Python library without the MCP server, who train or run models from the command line, who start the server by hand, or who run the test suite. Installing the server and connecting it to Claude is covered in the [README](../README.md).

## Library-only install

To use the Python library without the MCP server:

```bash
pip install git+https://github.com/sproft/tfbs-mcp.git
```

PyTorch is installed automatically. If you need a specific CUDA build, install torch first following <https://pytorch.org/>, then this package.

## Library usage

```python
import tfbs.nn.models as models
from tfbs.dataloaders.SeqDataset import TFBSDataModule

model = models.VCNNBpnet.load_from_checkpoint("path/to/best.ckpt")
data = TFBSDataModule(data_path="data/CTCF", batch_size=64)
```

## Training

```bash
tfbs-train fit --config my_config.yaml
```

`tfbs-train` is installed by pip, but the example configs live in the repo and are not packaged. Clone the repository to get `scripts/cli/config.example.yaml`, copy it, and replace the `/path/to/...` placeholders with your own data and checkpoint locations. The `tfbs_config` MCP tool generates a fresh config if you would rather start from scratch; it errors unless the `data_path` you give it already exists.

The scripts under `scripts/` read `TFBS_PROJECT_ROOT`, the directory holding `data/`, `saved_models/` and `results/`. It defaults to the repository root, so a fresh clone runs unedited. Set it if your data lives elsewhere:

```bash
export TFBS_PROJECT_ROOT=/mnt/lab/tfbs
```

## Prediction

```bash
tfbs-predict --peak_file peaks.bed --model_name my_model --config_file config.yml
```

## Running the server by hand

`tfbs-mcp` with no arguments starts the server and waits silently for an MCP client on standard input; press Ctrl+C to quit. `tfbs-mcp --check` verifies the install. `tfbs-mcp --setup` collects the reference-data paths, prints the Claude Code command and the Claude Desktop config block with them filled in, and can run the Claude Code command for you; both are described in the README.

To talk to the server without Claude, use the MCP Inspector:

```bash
npx -y @modelcontextprotocol/inspector "$(which tfbs-mcp)"
```

The reference-data variables (`TFBS_GENOME_FASTA`, `TFBS_MOTIF_FILE`, `TFBS_FOLDX_PDB`, `TFBS_CLINVAR_VCF`, `TFBS_PROJECT_ROOT`) are read from the server's environment, so export them in the shell before starting the Inspector.

## Optional extras

The install command in the README already includes everything the server needs. This table is only for choosing a different set:

| Extra | Installs | Needed for |
|---|---|---|
| `mcp` | the MCP protocol library | the server itself |
| `genomics` | pyfaidx, tangermeme | reading a genome FASTA |
| `viz` | captum, tangermeme, logomaker, seaborn | attribution, mutagenesis, seqlets |
| `tracking` | wandb | experiment logging during training |
| `foldx` | pyBigWig | not FoldX itself, see [Adding your data](../README.md#adding-your-data) in the README |
| `equinet` | equirc | the `EquiNet` architecture only |
| `dev` | pytest | running the test suite (also needs `mcp` and `viz`) |
| `all` | all of the above | |

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

`EquiNet` additionally needs the `equirc` package, which is on PyPI. Install it directly, or add the `equinet` extra to the install command from the README:

```bash
pip install equirc
```

Without it, constructing an `EquiNet` raises an `ImportError` naming that command. Every other architecture works without the extra.

## Demo data

`tfbs/data/demo/` ships a 10 kb GRCh38 slice around the thyroglobulin promoter, a BED of four NKX2-1 motif hits in it, and the JASPAR MA1994.1 motif. It is installed as package data, so `tfbs-mcp --setup --demo` works on a plain pip install. Provenance and coordinates are in `tfbs/data/demo/README.md`. To rebuild it from a FASTA that contains chromosome 8 (`NC_000008.11`), with `pyfaidx` installed:

```bash
python scripts/make_demo_data.py --fasta chr8.fa --record NC_000008.11 --start 132861953 --end 132871953 --source GCF_000001405.40 --meme MA1994.1.meme --out tfbs/data/demo
```

## Project structure

```
tfbs/                     Python package
├── nn/models.py          Model architectures (VCNNBpnet, RNN, CNN, MLP, BPNet)
├── dataloaders/          Data loading and one-hot encoding
├── prediction/           Window-based scoring and analysis
├── predict.py            ChIP-seq peak prediction CLI
├── cli/train_cli.py      Lightning CLI backing the tfbs-train command
├── data/demo/            Bundled demo genome slice, peaks and motif (package data)
└── mcp/                  MCP server
    ├── cli.py            tfbs-mcp command: version guard, --check and --setup
    ├── server.py         FastMCP server, 30 of 33 tools
    └── foldx_tools.py    Structure-based tools

scripts/                  Repo-only analysis scripts (not installed by pip)
├── cli/                  Training configs and SLURM job templates
├── make_demo_data.py     Rebuilds tfbs/data/demo/ from a GRCh38 FASTA
├── chipseq_benchmark*.py ChIP-seq classification benchmarks
├── disease_variant_validation.py
├── remap_validation.py   ReMap external validation
└── ...

docs/development.md       This page
docs/other-clients.md     Setup for AI apps other than Claude
tests/                    pytest suite
```

## Running tests

```bash
pip install ".[dev,mcp,viz,genomics]"
pytest
```

The `dev` extra is only pytest. The suite imports `tfbs.mcp.server`, so it needs `mcp` to collect at all, and the mutagenesis/marginalize tests exercise real tangermeme (from `viz` or `genomics`). Installing `dev` alone is not enough. Continuous integration runs the suite on Python 3.10 to 3.13 with a CPU-only PyTorch build, followed by `tfbs-mcp --check`.
