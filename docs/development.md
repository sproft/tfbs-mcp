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

`tfbs-train` and the example configs are both installed by pip. The configs live in the package under `tfbs/configs/`: `config.example.yaml`, one model block per architecture in `models/`, a trainer block in `trainers/`, and `wandb_sweep_config.yaml`, the search space `tfbs_sweep` can use. Print their installed location with:

```bash
python -c "import tfbs.configs as c; print(c.config_dir())"
```

Copy `config.example.yaml` from there and replace the `/path/to/...` placeholders with your own data and checkpoint locations. The `tfbs_config` MCP tool generates a fresh config from the same model templates if you would rather start from scratch; it errors unless the `data_path` you give it already exists.

The MCP tools resolve data, checkpoints and results against `TFBS_PROJECT_ROOT`, the directory holding `data/`, `saved_models/` and `results/`; the benchmark and validation pipelines run with it as their working directory. A config from `tfbs_config` trains to `saved_models/<dataset>/<scaling_method>/best_<model_type>.ckpt` (the tool reports it as `checkpoint_path`), and the benchmark tools look there; the pipelines take `--scaling-method` or explicit `--checkpoints NAME=PATH`. It defaults to the repository root in a clone and to the server's working directory otherwise. The SLURM helpers under `scripts/cli/` read it too. Set it if your data lives elsewhere:

```bash
export TFBS_PROJECT_ROOT=/mnt/lab/tfbs
```

## Training and validation pipelines

The six tools that run a longer job (`tfbs_validate`, `tfbs_sweep`, `tfbs_chipseq_benchmark`, `tfbs_chipseq_method_comparison`, `tfbs_remap_validation`, `tfbs_disease_variant_validation`) start a module from the installed package as `python -m tfbs.pipelines.<name>`, directly or inside a SLURM job, with the same interpreter as the server. Each one also runs by hand and lists its options with `--help`:

```bash
python -m tfbs.pipelines.remap_validation --help
```

The modules are `validate`, `run_sweep`, `chipseq_benchmark`, `chipseq_benchmark_with_foldx`, `remap_validation` and `disease_variant_validation`. In a clone, `scripts/<name>.py` (and `scripts/cli/run_sweep.py`) are thin wrappers around them, so older commands keep working, including `main(args)` with an `argparse.Namespace` for `validate` and `run_sweep`.

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
| `genomics` | pyfaidx, tangermeme, pysam (Linux and macOS; Windows has no pysam wheel) | reading a genome FASTA (`tfbs_extract_loci`, the benchmark and validation tools); pysam reads the ClinVar VCF in `tfbs_disease_variant_validation`, which therefore needs Linux, macOS or WSL |
| `viz` | captum, tangermeme, logomaker, seaborn | attribution, mutagenesis, seqlets |
| `tracking` | wandb | `tfbs_sweep`, and Weights & Biases logging during training |
| `foldx` | pyBigWig | the bigWig track of `tfbs_binding_scan`; not FoldX itself, see [Adding your data](../README.md#adding-your-data) in the README |
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

Each has a ready-to-use config in `tfbs/configs/models/`, installed with the package. Compose one with the full example config (paths as in a clone; after a pip install, use the directory printed under [Training](#training)):

```bash
tfbs-train fit --config tfbs/configs/config.example.yaml --model tfbs/configs/models/VCNNBpnet.yaml
```

`VCNNBpnet.yaml` and `RNN.yaml` carry the tuned settings from `do_train_final.sh`; the rest use class defaults. All thirteen instantiate at a 24 bp window and take one-hot `(batch, 4, L)` input. `FlexibleMLP` flattens that input and sizes its first layer from `input_length`, so its template sets `input_length: 96` (4 x 24) and `tfbs_config` sets it to 4 x the window. `MultiCNN` also needs a covariate tensor, as noted in its file.

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
├── pipelines/            Validation, sweep and benchmark jobs the MCP tools run
├── configs/              Training config templates (package data)
├── data/demo/            Bundled demo genome slice, peaks and motif (package data)
└── mcp/                  MCP server
    ├── cli.py            tfbs-mcp command: version guard, --check and --setup
    ├── server.py         FastMCP server, 30 of 33 tools
    └── foldx_tools.py    Structure-based tools

scripts/                  Repo-only scripts (not installed by pip)
├── cli/                  SLURM job templates; run_sweep.py wraps tfbs.pipelines
├── make_demo_data.py     Rebuilds tfbs/data/demo/ from a GRCh38 FASTA
├── chipseq_benchmark*.py Wrappers around tfbs.pipelines (likewise
├── remap_validation.py   validate.py and disease_variant_validation.py)
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
