# tfbs-mcp

A Python library and Model Context Protocol (MCP) server for transcription factor binding site analysis. The library packages neural network architectures, ChIP-seq data pipelines, attribution and motif analysis, and structure-based binding-energy scoring. The MCP server exposes 33 of these operations as tools that a large language model client can call directly, so an LLM can compose a full TFBS analysis from a natural-language goal.

## Installation

```bash
# Library only (no MCP server)
pip install git+https://github.com/sproft/tfbs-mcp.git

# To run the MCP server, the [mcp] extra is required
pip install "tfbs-mcp[mcp,genomics,viz] @ git+https://github.com/sproft/tfbs-mcp.git"

# Everything
pip install "tfbs-mcp[all] @ git+https://github.com/sproft/tfbs-mcp.git"
```

The first command installs the library but **not** the MCP protocol package, so
the `tfbs-mcp` command would fail with `ModuleNotFoundError: No module named
'mcp'`. Include the `mcp` extra whenever you intend to run the server.

PyTorch is a hard dependency and pip installs a default build automatically. If
you need a specific CUDA version, install torch first per
<https://pytorch.org/>, then install this package.

New to MCP? See [Setting up the MCP server](#setting-up-the-mcp-server-beginner-guide)
for a step-by-step walkthrough.

## Setting up the MCP server (beginner guide)

An **MCP server** is a small program that sits on your computer and offers a set
of tools to an AI assistant. It does not talk to the internet and it has no
window — the AI app starts it for you in the background, then can call
`tfbs_predict`, `tfbs_fimo` and the other 31 tools while answering you.
"Connecting" means writing one small config file that tells your AI app where
the `tfbs-mcp` program lives on disk.

This server speaks **stdio** only — the AI app launches it as a subprocess. It
does not listen on a port, so any client that needs a URL requires a bridge
(see [OpenAI and Google](#openai-and-google)).

Budget 20 minutes. Steps 1–3 are the same for everyone; step 4 depends on
which AI app you use.

### Step 1 — Install Python and create a virtual environment

A *virtual environment* ("venv") is a private folder for this project's Python
packages, so installing them cannot break anything else on your machine.

**Opening a terminal:** on Windows press the Windows key, type `powershell`,
and click **Windows PowerShell** — a window opens with a prompt like
`PS C:\Users\you>`. On macOS press Cmd+Space, type `terminal`, press Enter.
You type one line, press Enter, and wait for the prompt to come back before
typing the next.

First check your Python:

```powershell
python --version
```

Three things can happen:

- It prints `Python 3.10` or higher — good, continue.
- It prints `Python 3.9` or lower, **or** `'python' is not recognized`, **or**
  the Microsoft Store opens — install Python 3.11 from
  <https://www.python.org/downloads/>, tick **"Add python.exe to PATH"** in the
  installer, then close and reopen PowerShell and check again.
- On macOS/Linux, try `python3 --version` and use `python3` everywhere below.

Now create the venv. Windows PowerShell:

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

Your prompt should now start with `(tfbs-env)`. That prefix is the whole point
— it means the venv is active.

> **If PowerShell says `Activate.ps1 cannot be loaded because running scripts
> is disabled`,** run this once, answer `Y`, then run the activation line
> again:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

> **The venv only lasts as long as that terminal window.** Next time you open a
> new terminal you must re-run the activation line before `pip` or `tfbs-mcp`
> will work. Your AI app does *not* need the venv activated — that is exactly
> why step 3 uses a full absolute path.

### Step 2 — Install tfbs-mcp

**The `[mcp]` part is not optional.** Without it pip installs the library but
not the MCP protocol package, and the `tfbs-mcp` command will crash with
`ModuleNotFoundError: No module named 'mcp'`.

```bash
pip install "tfbs-mcp[mcp,genomics,viz] @ git+https://github.com/sproft/tfbs-mcp.git"
```

This downloads PyTorch — roughly **2–3 GB, 5–20 minutes**. It is not stuck.
Do not close the window.

Then verify in the terminal, *before* touching any AI app:

```bash
python -c "import tfbs, mcp; print('ok')"
```

If that prints `ok`, the install worked.

| Extra | Installs | Needed for |
|---|---|---|
| `mcp` | the MCP protocol library | **the server itself — always include it** |
| `genomics` | pyfaidx, tangermeme | reading a genome FASTA |
| `viz` | captum, tangermeme, logomaker, seaborn | attribution, mutagenesis, seqlets |
| `tracking` | wandb | experiment logging during training |
| `foldx` | pyBigWig | *not* FoldX itself — see [What needs extra setup](#what-needs-extra-setup) |
| `equinet` | equirc | the `EquiNet` architecture only |
| `dev` | pytest | running the test suite (also needs `mcp` and `viz`) |

### Step 3 — Find the absolute path to `tfbs-mcp`

**This is the step people get wrong.** Your AI app is not launched from your
terminal, so it does not know about your venv and a bare `tfbs-mcp` will not
work. Every config below needs the full path.

Windows PowerShell:

```powershell
(Get-Command tfbs-mcp).Source
```

macOS / Linux:

```bash
which tfbs-mcp
```

You should see something like `C:\Users\jsmith\tfbs-env\Scripts\tfbs-mcp.exe`
or `/Users/jsmith/tfbs-env/bin/tfbs-mcp`. Copy it somewhere — you will paste it
in step 4. If instead you get a red `The term 'tfbs-mcp' is not recognized`,
your venv is not active: go back and re-run the activation line from step 1.

> **Every example below shows `C:\Users\you\...` or `/Users/you/...`. That is a
> placeholder.** Substitute the string you just copied. Pasting the examples
> unchanged is the single most common failure.

Three rules for the JSON files in step 4:

1. **On Windows, double every backslash.** `C:\Users\you` must be written
   `C:\\Users\\you`. A single backslash makes the file invalid and the server
   silently never appears. (Forward slashes also work: `C:/Users/you`.)
2. **If the file already exists, merge — do not append.** A JSON file has
   exactly one outer `{ }`. If `mcpServers` is already there, add your `"tfbs"`
   block *inside* it and put a comma after the previous entry.
3. **The folder often does not exist yet.** Most of these live in folders
   starting with a dot, which Windows Explorer refuses to create. Make them
   from the terminal, e.g. `mkdir $HOME\.gemini`.

### Step 4 — Point your AI app at the server

Find your app below. The server registers itself under the name **`tfbs`**.

#### Anthropic

**Claude Desktop** — open **Settings → Developer → Edit Config**. That button
creates the file if it is missing, which is more reliable than hunting for it
(`%APPDATA%\Claude\claude_desktop_config.json` on Windows,
`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS).

```json
{
  "mcpServers": {
    "tfbs": {
      "command": "C:\\Users\\you\\tfbs-env\\Scripts\\tfbs-mcp.exe",
      "args": [],
      "env": {}
    }
  }
}
```

**Fully quit and reopen Claude Desktop** — closing the window is not enough.

**Claude Code** (CLI) — no file editing needed. Note the `--` before the path;
leaving it out is the most common mistake:

```bash
claude mcp add tfbs --scope user -- "C:\Users\you\tfbs-env\Scripts\tfbs-mcp.exe"
```

This needs the `claude` CLI installed separately. The Claude Code VS Code
extension shares this same configuration — add servers from its integrated
terminal, then manage them with `/mcp` in the chat panel.

#### Code editors

| Editor | File | Top-level key |
|---|---|---|
| VS Code (Copilot agent mode) | `.vscode/mcp.json` in your project | **`servers`** |
| Cursor | `~/.cursor/mcp.json`, or `.cursor/mcp.json` per project | `mcpServers` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` (global only) | `mcpServers` |
| Cline | open via sidebar → **MCP Servers → Configure** | `mcpServers` |
| Zed | `settings.json` via **"zed: open settings file"** | **`context_servers`** |

VS Code is the odd one out — it uses `servers`, not `mcpServers`. Pasting a
Cursor snippet into VS Code silently does nothing:

```json
{
  "servers": {
    "tfbs": {
      "type": "stdio",
      "command": "C:\\Users\\you\\tfbs-env\\Scripts\\tfbs-mcp.exe",
      "args": []
    }
  }
}
```

Cursor, Windsurf and Cline use the Claude Desktop shape shown above. Zed nests
its entry under `context_servers` instead. **Continue** is different again —
its `~/.continue/config.yaml` takes a *list*, and each entry needs a `name`:

```yaml
mcpServers:
  - name: tfbs
    command: C:\Users\you\tfbs-env\Scripts\tfbs-mcp.exe
    args: []
```

#### OpenAI and Google

Some OpenAI products run local stdio servers and some cannot. The difference
matters:

| Product | Works with this server? |
|---|---|
| **OpenAI Codex** (CLI / IDE) | Yes — directly, stdio |
| **Gemini CLI** | Yes — directly, stdio |
| **OpenAI Agents SDK** (Python) | Yes — directly, in your own script |
| **ChatGPT** app, **Responses API** | No — needs a bridge (below) |

**OpenAI Codex** — `~/.codex/config.toml` (TOML, not JSON):

```toml
[mcp_servers.tfbs]
command = "C:\\Users\\you\\tfbs-env\\Scripts\\tfbs-mcp.exe"
args = []
```

**Gemini CLI** — `~/.gemini/settings.json`, standard `mcpServers` shape:

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

**OpenAI Agents SDK** — no config file; you launch the server from Python:

```python
import asyncio
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

TFBS = r"C:\Users\you\tfbs-env\Scripts\tfbs-mcp.exe"

async def main():
    async with MCPServerStdio(
        name="tfbs",
        params={"command": TFBS, "args": []},
        client_session_timeout_seconds=120,
    ) as server:
        agent = Agent(name="TFBS", instructions="Use the tfbs tools.",
                      mcp_servers=[server])
        result = await Runner.run(agent, "What environment is the TFBS server running in?")
        print(result.final_output)

asyncio.run(main())
```

**ChatGPT and the Responses API** cannot launch a local program — OpenAI's
servers make the call, so the server must be reachable over HTTPS. Bridge it:

```bash
npx -y supergateway --stdio "C:/Users/you/tfbs-env/Scripts/tfbs-mcp.exe" --outputTransport streamableHttp --port 8000
```

That serves `http://localhost:8000/mcp`. To reach it from ChatGPT you must also
expose it publicly (e.g. `cloudflared tunnel --url http://127.0.0.1:8000`) and
paste the resulting HTTPS URL into ChatGPT's connector settings. **Think before
you do this** — it puts a tool that reads local files behind a public URL.

#### Local and open-source models

| Runner | How |
|---|---|
| **LM Studio** | Sidebar → **Program → Install → Edit mcp.json**; `mcpServers` shape. Needs ≥ 0.3.17 |
| **Jan** | **Settings → MCP Servers → Add Server**; fill in Command with your path |
| **LibreChat** | `mcpServers:` block in `librechat.yaml` |
| **Open WebUI** | Wrap with `mcpo`, then add the URL under Settings → Integrations → Tools |
| **Ollama** | No native MCP. Use a client in front of it, e.g. `ollmcp` |

Ollama itself does not speak MCP — it provides an OpenAI-compatible endpoint
and tool calling, but something must act as the MCP client:

```bash
pip install --upgrade ollmcp
ollmcp --servers-json tfbs-servers.json --model qwen3:8b
```

Open WebUI needs the stdio server turned into an OpenAPI service first:

```bash
uvx mcpo --port 8000 --api-key "choose-a-secret" -- "C:/Users/you/tfbs-env/Scripts/tfbs-mcp.exe"
```

Then add `http://localhost:8000` as a Tool in Open WebUI.

### Step 5 — Check it worked

Restart your AI app (Claude Desktop needs a full quit, not just closing the
window). Then ask it:

> Check the TFBS server environment — is it running on SLURM, does it see a
> GPU, and what models are loaded?

That calls `tfbs_server_info` and `tfbs_list_models`, which need no data files
at all. You should get back a small JSON blob reporting `on_slurm`,
`gpu_count`, `default_device` and an empty model list.

Two things to expect:

- **The first tool call is slow — allow up to a minute.** The server defers
  loading PyTorch so it can connect instantly, and the first real call pays that
  cost (longer on Windows, or the first time after a reboot). Later calls are
  fast. It is not frozen.
- **Errors come back as normal-looking JSON**, e.g.
  `{"error": "...", "fix": "..."}`, not as a red failure. If a tool seems to
  "work" but the answer looks wrong, ask to see the raw response and check for
  an `error` key.

If you would rather test without any AI app, the official MCP Inspector talks
to the server directly:

```bash
npx -y @modelcontextprotocol/inspector "C:/Users/you/tfbs-env/Scripts/tfbs-mcp.exe"
```

### Troubleshooting

| What you see | Why | Fix |
|---|---|---|
| Server missing from the app; no error | Invalid JSON — usually a single backslash on Windows, or a trailing comma | Double the backslashes; paste the file into <https://jsonlint.com> |
| `ModuleNotFoundError: No module named 'mcp'` | Installed without the `[mcp]` extra | `pip install "mcp>=1.0"` into the same venv |
| `The term 'tfbs-mcp' is not recognized` | venv not active, or install failed | Re-run the activation line from step 1 |
| Server shows as "failed" / disconnects at startup | Wrong path, or a relative path instead of an absolute one | Re-run step 3 and paste the exact output |
| VS Code shows nothing, other editors fine | Used `mcpServers` instead of `servers` in `.vscode/mcp.json` | Rename the key to `servers` |
| Works in the terminal, not in the app | The app does not inherit your venv | Use the absolute path — never a bare `tfbs-mcp` |
| `{"error": "No reference genome FASTA configured."}` | The tool needs data you have not pointed it at | Pass the path as an argument, or set `TFBS_GENOME_FASTA` (below) |
| Times out on startup | Slow disk or antivirus scanning the venv | Claude Code: set `MCP_TIMEOUT=60000` before launching |

If nothing else works, run the executable directly in your terminal. A healthy
server prints nothing and appears to hang — that is correct, it is waiting for
input. Press Ctrl+C. Any traceback you see instead is the real problem.

### What needs extra setup

Only **5 of the 33 tools work with nothing but the install**:
`tfbs_server_info`, `tfbs_list_models`, `tfbs_preprocess`,
`tfbs_classify_metrics`, and `tfbs_fetch_structure` (which needs internet).
Note that `tfbs_config` is *not* one of them — it errors unless the
`data_path` you give it already exists. The rest need data or software you
must supply:

| To use | You need | Point it there with |
|---|---|---|
| Sequence extraction, benchmarks | A reference genome FASTA (hg38 is ~3 GB) | `TFBS_GENOME_FASTA` |
| `tfbs_fimo`, motif scanning | The `fimo` binary from the [MEME Suite](https://meme-suite.org/) on your PATH, plus a JASPAR/MEME motif file | `TFBS_MOTIF_FILE` |
| Structure-based scoring | **FoldX** — a separately licensed commercial binary from <https://foldxsuite.crg.eu/>, plus a repaired TF–DNA PDB. The `foldx` pip extra installs pyBigWig, *not* FoldX | `TFBS_FOLDX_PDB` |
| Disease-variant validation | An indexed ClinVar VCF (`.vcf.gz` + `.tbi`) | `TFBS_CLINVAR_VCF` |
| Prediction, attribution | A trained model checkpoint — none ship with the package | `tfbs_load_model` |
| `slurm_*`, and any `slurm=true` option | An HPC cluster with SLURM | — |
| Training, benchmark, validation tools | A git clone (they run scripts under `scripts/`, which pip does not install) | `TFBS_PROJECT_ROOT`, else the directory containing the installed `tfbs` package |

Set the environment variables in the `env` block of your client config, so the
tools find your data without you repeating the path every time:

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

Any tool argument you pass explicitly overrides these. If neither is set, the
tool tells you which argument and which variable would fix it.

The repo-only training scripts and benchmark scripts under `scripts/` use one
further variable, `TFBS_PROJECT_ROOT`, for the directory holding `data/`,
`saved_models/` and `results/`. The MCP tools that shell out to those scripts
read it too, falling back to the directory containing the installed `tfbs`
package. It defaults to the repo root, so a fresh clone runs unedited; set it
if your data or your checkout lives elsewhere:

```bash
export TFBS_PROJECT_ROOT=/mnt/lab/tfbs
```

The example YAML configs under `scripts/` contain `/path/to/...` placeholders
rather than real paths — copy one and fill in your own before using it.

## Quick start

### MCP server

```bash
tfbs-mcp                     # stdio transport for Claude Code / VS Code
```

The server registers 33 tools across six functional layers (inference, SLURM management, training, data pipeline, structure-based validation, validation/comparison), and identifies itself to clients as `tfbs`. stdio is the only transport.

Point your client at the absolute path of the `tfbs-mcp` executable — see
[Setting up the MCP server](#setting-up-the-mcp-server-beginner-guide) for
per-client config files. Note that most tools need a genome FASTA, MEME Suite,
FoldX or a trained checkpoint before they do anything useful.

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

`tfbs-train` is installed by pip, but the example configs live in the repo and
are not packaged — clone the repository to get `scripts/cli/config.example.yaml`,
copy it, and replace the `/path/to/...` placeholders with your own data and
checkpoint locations. The `tfbs_config` MCP tool generates a fresh config if you
would rather start from scratch.

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
- **SimpleCNN**, **CNN**, **FlexibleCNN**, **VCNN**, **my_cnn**: Convolutional variants
- **MLP**, **FlexibleMLP**: Fully connected networks
- **BPNet**, **BPNetReal**: BPNet architectures for base-resolution count data
- **MultiCNN**: CNN that also consumes numerical covariates
- **EquiNet**: Reverse-complement equivariant network (needs the `equinet` extra)

Each has a ready-to-use config in `scripts/cli/yamls/models/`. Compose one with
the full example config:

```bash
tfbs-train fit --config scripts/cli/config.example.yaml --model scripts/cli/yamls/models/VCNNBpnet.yaml
```

`VCNNBpnet.yaml` and `RNN.yaml` carry the tuned settings from
`do_train_final.sh`; the rest use class defaults. All thirteen instantiate at a
24 bp window, and eleven accept one-hot `(batch, 4, L)` input directly —
`FlexibleMLP` expects tabular features and `MultiCNN` also needs a covariate
tensor, as noted in those files.

`EquiNet` additionally needs the optional `equirc` package. `tfbs-mcp` is not
on PyPI, so install the dependency directly (it *is* on PyPI):

```bash
pip install equirc
```

Without it, constructing an `EquiNet` raises an `ImportError` naming that
command. Every other architecture works without the extra.

## Project structure

```
tfbs/                     Python package
├── nn/models.py          Model architectures (VCNNBpnet, RNN, CNN, MLP, BPNet)
├── dataloaders/          Data loading and one-hot encoding
├── prediction/           Window-based scoring and analysis
├── predict.py            ChIP-seq peak prediction CLI
├── cli/train_cli.py      Lightning CLI backing the tfbs-train command
└── mcp/                  MCP server
    ├── server.py         FastMCP server, 30 of 33 tools
    └── foldx_tools.py    Structure-based tools

scripts/                  Repo-only analysis scripts (not installed by pip)
├── cli/                  Training configs and SLURM job templates
├── chipseq_benchmark*.py ChIP-seq classification benchmarks
├── disease_variant_validation.py
├── remap_validation.py   ReMap external validation
└── ...

tests/                    pytest suite
```

## Running tests

```bash
pip install ".[dev,mcp,viz,genomics]"
pytest
```

The `dev` extra is only pytest. The suite imports `tfbs.mcp.server`, so it needs
`mcp` to collect at all, and the mutagenesis/marginalize tests exercise real
tangermeme (from `viz` or `genomics`). Installing `dev` alone is not enough.

## Citation

If you use `tfbs-mcp` in your work, please cite:

> Proft S, Leser U. tfbs-mcp: a Model Context Protocol server for agentic transcription factor binding site analysis. *Bioinformatics*, 2026 (in preparation).

## License

MIT. See [LICENSE](LICENSE) for details.
