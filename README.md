# tfbs-mcp

A Python library and Model Context Protocol (MCP) server for transcription factor binding site analysis. The library packages neural network architectures, ChIP-seq data pipelines, attribution and motif analysis, and structure-based binding-energy scoring. The MCP server exposes 33 of these operations as tools that a large language model client can call directly, so an LLM can compose a full TFBS analysis from a natural-language goal.

## Installation

```bash
# Minimal library install
pip install git+https://github.com/sproft/tfbs-mcp.git

# With the MCP server and FoldX (structure-based scoring)
pip install "tfbs-mcp[mcp,foldx] @ git+https://github.com/sproft/tfbs-mcp.git"

# Everything
pip install "tfbs-mcp[all] @ git+https://github.com/sproft/tfbs-mcp.git"
```

PyTorch should be installed separately per <https://pytorch.org/> for your CUDA version.

## Quick start

### MCP server

```bash
tfbs-mcp                     # stdio transport for Claude Code / VS Code
```

The server registers 33 tools across six functional layers (inference, SLURM management, training, data pipeline, structure-based validation, validation/comparison). Configure your MCP-compatible client to connect to the `tfbs-mcp` entry point.

### Library

```python
import tfbs.nn.models as models
from tfbs.dataloaders.SeqDataset import TFBSDataModule

model = models.VCNNBpnet.load_from_checkpoint("path/to/best.ckpt")
data = TFBSDataModule(data_dir="data/CTCF", batch_size=64)
```

### Training

```bash
tfbs-train fit --config scripts/cli/config.yaml
```

### Prediction

```bash
tfbs-predict --peak_file peaks.bed --model_name my_model --config_file config.yml
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
- **SimpleCNN**, **CNN**, **FlexibleCNN**: Convolutional variants
- **MLP**, **FlexibleMLP**: Fully connected networks
- **BPNet**, **BPNetReal**: BPNet architectures for base-resolution count data

## Project structure

```
tfbs/                     Python package
├── nn/models.py          Model architectures (VCNNBpnet, RNN, CNN, MLP, BPNet)
├── dataloaders/          Data loading and one-hot encoding
├── prediction/           Window-based scoring and analysis
├── predict.py            ChIP-seq peak prediction CLI
└── mcp/                  MCP server
    ├── server.py         FastMCP server, 30 of 33 tools
    └── foldx_tools.py    Structure-based tools

scripts/                  Training, analysis, and visualization scripts
├── cli/                  Lightning CLI training with SLURM job templates
├── chipseq_benchmark*.py ChIP-seq classification benchmarks
├── disease_variant_validation.py
├── remap_validation.py   ReMap external validation
└── ...

tests/                    pytest suite
```

## Running tests

```bash
pip install ".[dev]"
pytest
```

## Citation

If you use `tfbs-mcp` in your work, please cite:

> Proft S, Leser U. tfbs-mcp: a Model Context Protocol server for agentic transcription factor binding site analysis. *Bioinformatics*, 2026 (in preparation).

## License

MIT. See [LICENSE](LICENSE) for details.
