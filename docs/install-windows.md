# Installing on Windows

tfbs-mcp runs on Windows. Follow the installation steps in the [README](../README.md#installation) in Windows PowerShell. This page lists the places where Windows differs, and the three tools that need WSL. `tfbs-mcp --check`, `tfbs-mcp --setup` and the prompts are the same.

## Step 1: Create an environment

The micromamba install guide linked from the README (<https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>) covers Windows as well. Once micromamba is installed, the `micromamba` lines of step 1 are the same.

Creating and activating a venv uses `python` instead of `python3` and a different activation line:

```powershell
cd $HOME
python -m venv tfbs-env
tfbs-env\Scripts\Activate.ps1
```

If PowerShell says `Activate.ps1 cannot be loaded because running scripts is disabled`, run the following once, answer `Y`, and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

In a new PowerShell window, run the activation line again before using `pip` or `tfbs-mcp`.

## Step 3: Connect Claude

PowerShell has no `which`. Wherever the README writes `"$(which tfbs-mcp)"`, use `(Get-Command tfbs-mcp).Source`:

```powershell
claude mcp add tfbs --scope user -- (Get-Command tfbs-mcp).Source
```

The same goes for the [By hand](../README.md#by-hand) line with `--env` and for the SLURM line of the README.

In `claude_desktop_config.json`, every backslash in a Windows path is written twice, as in `C:\\Users\\you\\...`. The blocks printed by `tfbs-mcp --check` and `tfbs-mcp --setup` already have them, so paste those rather than typing a path.

## Other commands

- **FoldX:** rename the program to `foldx.exe` and put it in the `Scripts` folder of the environment, next to the `tfbs-mcp.exe` that `tfbs-mcp --check` prints.
- **Uninstall a venv:** `Remove-Item -Recurse -Force $HOME\tfbs-env` instead of `rm -rf ~/tfbs-env`.

## Tools that need WSL

Three tools need programs that do not exist for Windows, so they need Linux, macOS or WSL:

- `tfbs_disease_variant_validation` reads the ClinVar VCF with pysam, which has no Windows wheel, so the `genomics` extra leaves pysam out on Windows and `tfbs-mcp --check` prints a note about it.
- `tfbs_remap_validation` and `tfbs_chipseq_method_comparison` run the MEME Suite's `fimo` program for their FIMO leg, and the MEME Suite has no native Windows build. They have no fallback and stop with an error naming `fimo` before anything runs. The ClinVar tool runs `fimo` too, but its FIMO leg is skipped when no motif file is given and `TFBS_MOTIF_FILE` is unset.

Every other tool runs natively. `tfbs_fimo` is the only tool that falls back to tangermeme's FIMO reimplementation when the `fimo` program is missing.

A single tool cannot run under WSL on its own: the whole server has to. To set that up:

1. **Install inside WSL.** Install WSL (<https://learn.microsoft.com/windows/wsl/install>; `wsl --install` installs Ubuntu), open its terminal, and follow steps 1 and 2 of the README there, with the Linux commands. Add the MEME Suite to the environment (`micromamba install -c bioconda -c conda-forge meme`) and run `tfbs-mcp --check`. Your Windows files are under `/mnt/c/...` inside WSL.
2. **Connect Claude.** The simplest way is to install Claude Code inside WSL as well and follow the README from step 3 on; everything then behaves as on Linux. To keep using Claude on Windows, start the server through `wsl.exe` instead. Variables registered on the Windows side do not reach WSL, so pass the data paths through `env`, with WSL paths. For Claude Code:

   ```powershell
   claude mcp add tfbs --scope user -- wsl.exe -e env TFBS_GENOME_FASTA=/mnt/c/data/hg38.fa TFBS_MOTIF_FILE=/mnt/c/data/MA1994.1.meme /home/you/micromamba/envs/tfbs-mcp/bin/tfbs-mcp
   ```

   For Claude Desktop, the `tfbs` entry becomes `"command": "wsl.exe"` with `"args": ["-e", "env", "TFBS_GENOME_FASTA=/mnt/c/data/hg38.fa", "/home/you/micromamba/envs/tfbs-mcp/bin/tfbs-mcp"]`. Use the path that `tfbs-mcp --check` prints inside WSL.
3. **Check it.** Restart Claude and send the test prompt of step 4 of the README. If `tfbs` does not connect, run the `wsl.exe -e ...` command above by hand in PowerShell: it should wait silently (Ctrl+C to quit); an error there names the problem.

Problems and their fixes are listed in [troubleshooting.md](troubleshooting.md).
