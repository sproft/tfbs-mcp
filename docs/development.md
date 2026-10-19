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

`tfbs-train` and the example configs are both installed by pip. The configs live in the package under `tfbs/configs/`: `config.example.yaml`, one model block per architecture in `models/`, a trainer block in `trainers/`, and `wandb_sweep_config.yaml`, the search space `tfbs_sweep` can use. Its agent (`python -m tfbs.pipelines.run_sweep`) trains RNN, VCNNBpnet and BPNetReal from the keys of that file. Print their installed location with:

```bash
python -c "import tfbs.configs as c; print(c.config_dir())"
```

Copy `config.example.yaml` from there and replace the `/path/to/...` placeholders with your own data and checkpoint locations. The `tfbs_config` MCP tool generates a fresh config from the same model templates if you would rather start from scratch; it errors unless the `data_path` you give it already exists.

The MCP tools resolve data, checkpoints and results against `TFBS_PROJECT_ROOT`, the directory holding `data/`, `saved_models/` and `results/`; the benchmark and validation pipelines run with it as their working directory. A config from `tfbs_config` trains to `saved_models/<dataset>/<scaling_method>/best_<model_type>.ckpt` (the tool reports it as `checkpoint_path`), and the benchmark tools look there; the pipelines take `--scaling-method` or explicit `--checkpoints NAME=PATH`. It defaults to the repository root in a clone and to the server's working directory otherwise. The SLURM helpers under `scripts/cli/` read it too. Set it if your data lives elsewhere:

```bash
export TFBS_PROJECT_ROOT=/mnt/lab/tfbs
```

### Where the tools find checkpoints

`TFBS_PROJECT_ROOT` is the directory holding `data/`, `saved_models/` and `results/`. The benchmark and validation tools run from it, so a relative `models_dir`, such as their default `saved_models_final`, is looked up there. A dataset's checkpoint is `<models_dir>/<dataset>/<scaling_method>/best_<model_type>.ckpt`, the layout `tfbs_config` trains to, with `scaling_method` defaulting to `standardize`; when `scaling_method` is not given and that file is missing, the one `<dataset>/*/best_<model_type>.ckpt` is used if there is exactly one, with a warning in the result. An explicit `scaling_method` never falls back to another folder, and the tools also take explicit `checkpoints` (`tfbs_disease_variant_validation` takes `nn_models`). The lookup is all-or-nothing: if any dataset has no single checkpoint, the class a Lightning checkpoint was saved with cannot be imported from the project root, or an inference checkpoint's `arch` is not `model_type`, nothing runs. Without the variable the server uses the repository root when it runs from a clone, and otherwise the directory it was started in; `tfbs_server_info` reports the one in use as `project_root`. Two checkpoint formats are accepted, by `tfbs_load_model` and by every pipeline (ReMap, ClinVar, both ChIP-Atlas benchmarks and `tfbs_validate`), all of which load through one function, `tfbs.nn.loading.load_checkpoint`, that tells them apart by content. A **Lightning checkpoint** is a `.ckpt` from training, loaded with `<class>.load_from_checkpoint`; the class module it records in `hyper_parameters["_class_path"]` is imported first. An **inference checkpoint** is a dict saved with `torch.save` that holds `arch` (a class in `tfbs.nn.models`), `hyper_parameters` (its constructor arguments), `state_dict`, and optionally `scaling` (`{scaling_method, mean, std}`), `library` and `metadata`; it is read with `weights_only=True`, rebuilt from its `hyper_parameters` and loaded with `strict=True`. The NKX2-1 study's released `<models>/flank/vcnnbpnet.pt` (and its `core/` and `all/` siblings, doi:10.5281/zenodo.21399578) are inference checkpoints. A bare state_dict (a file written by `torch.save(model.state_dict())`, a dict of tensors only) is neither, because it records no architecture, and neither is a file `torch.load` cannot read that is not a zip archive (an empty or truncated file); `load_checkpoint` raises a `ValueError` naming the file for both, and a pipeline exits with one line, `Cannot load <path>: <error>`, for any checkpoint it cannot load. The lookup above finds `best_<model_type>.ckpt` only; name a released file through `checkpoints` (`--checkpoints NAME=PATH`), `nn_models` (`--models NAME=PATH`) or `checkpoint_path`. The tools check an inference checkpoint's `arch`, which must be one of the model classes `tfbs_load_model` lists, and refuse a bare state_dict before anything runs, and skip the class-path probe for both. The pipelines score every model by its raw output, so an inference checkpoint is scored on its training scale (its `scaling` block is not applied), as a Lightning checkpoint is; this changes no AUC. `tfbs_load_model` applies the scaling block, so `tfbs_predict` reports label units.

The `tfbs_train` tool submits the training to SLURM by default. With `local=true` it runs `python -m tfbs.cli.train_cli fit --config <config> --data.num_workers 0` on the server's machine instead, from the config's directory, and blocks until training ends (at most `local_timeout` seconds, default 3600), so it is meant for small datasets and laptops. Its output goes to `<config stem>_train.log` next to the config. The trainer runs in a process group of its own, and the whole group is killed when `local_timeout` runs out or the MCP call is cancelled. The result reports the checkpoint and, from the `metrics.csv` this run's CSVLogger wrote (every `tfbs_config` config without `wandb_project` has one, under the checkpoint directory), the validation metrics of the best epoch of this run, the one with the lowest `val_loss` that ModelCheckpoint keeps, with `best_epoch` and `epochs_run`. `metrics` is null, with a `note` saying why, when the config has no CSVLogger or the run wrote no fresh `metrics.csv`; `val_auc` is null, with a note, when the validation split holds a single class. For a classifier, a `warning` says the model did not learn (constant output) when `val_loss` lies within 0.005 of ln 2 at every logged epoch, or when the best epoch is the first, a later epoch was logged and the best `val_loss` lies within 0.005 of ln 2, and, when `val_auc` is logged, the best epoch's `val_auc` also lies within 0.05 of 0.5. The AUC condition keeps a short run that is learning from being called dead: the three-layer `VCNNBpnet` default can stay within 0.005 of ln 2 for its first 1 to 3 epochs while its `val_auc` already reaches 0.6 to 0.8. A `val_auc` that is null because the validation split holds one class does not count. The warning suggests another seed (`extra_args ['--seed_everything 7']`), more data, or a shallower stack through `tfbs_config`'s `model_params` (for a `VCNNBpnet` with the study's five layers, the template default `{"dilations": [1, 1, 2]}`). Retraining into the same folder overwrites `best_<model_type>.ckpt`.

A SLURM job from a config (`config_path` without `local`) runs the same trainer from the config's directory. Its generated script requests `--cpus-per-task` from `cpus_per_task`, by default the config's `data.num_workers` plus 1 for the main training process (25 for the 24 workers of a `tfbs_config` config; an `extra_args` `--data.num_workers N` overrides the worker count), and writes a `--mem` line only when `mem` is given (e.g. `"64G"`; at least 1, since SLURM reads `--mem=0` as all of the node's memory), so otherwise the cluster's default memory applies. The result names `config_path` and `checkpoint_path`, where the job's ModelCheckpoint will write, derived from the config as in local mode; after the job, `slurm_logs` finds its `<job name>_<job id>.out` and `.err` in the config's directory even when sacct reports them as relative or unexpanded patterns. A relative path reported without a WorkDir is skipped rather than read from the server's own working directory.

## Training and validation pipelines

The six tools that run a longer job (`tfbs_validate`, `tfbs_sweep`, `tfbs_chipseq_benchmark`, `tfbs_chipseq_method_comparison`, `tfbs_remap_validation`, `tfbs_disease_variant_validation`) start a module from the installed package as `python -m tfbs.pipelines.<name>`, directly or inside a SLURM job, with the same interpreter as the server. Each one also runs by hand and lists its options with `--help`:

```bash
python -m tfbs.pipelines.remap_validation --help
```

In direct mode (`slurm=false`) the benchmark and validation tools run the module in a worker thread, so the server keeps answering other calls while it runs, and a cancelled call kills the module's whole process group (the process tree on Windows). `remap_validation`, `chipseq_benchmark_with_foldx` (behind `tfbs_chipseq_method_comparison`), and `disease_variant_validation` when it has a motif file, run the MEME Suite's `fimo` program, which must be on PATH (it has no native Windows build, see [install-windows.md](install-windows.md#tools-that-need-wsl)); the tools check for it before they run or submit anything. The same three score a structure with FoldX when they are given one (`--foldx-pdb`, which the tools fill from `foldx_pdb` or `TFBS_FOLDX_PDB`). They then need the FoldX binary, which they take from `--foldx-bin` (`remap_validation` only), else `TFBS_FOLDX_BIN`, else `foldx` or `foldx5` on PATH (on the compute node for a SLURM job), and without it they stop at startup, before anything is scored, with the FoldX-missing error, which suggests `skip_foldx=true`. Called with `skip_foldx=true`, the tools pass no structure (`remap_validation` gets `--skip-foldx`), so the FoldX leg is left out and FoldX is not needed. The modules are `validate`, `run_sweep`, `chipseq_benchmark`, `chipseq_benchmark_with_foldx`, `remap_validation` and `disease_variant_validation`. In a clone, `scripts/<name>.py` (and `scripts/cli/run_sweep.py`) are thin wrappers around them, so older commands keep working, including `main(args)` with an `argparse.Namespace` for `validate` and `run_sweep`. The peak BEDs of `tfbs_extract_loci`, `remap_validation` and `disease_variant_validation` are read through `tfbs.bedio`, which skips blank lines and lines that start with `#`, `track` or `browser`, keeps a `#` inside a line as data (ChIP-Atlas peak BEDs hold catalog numbers such as `Catalog#: MS-699-P` in column 4), and decompresses gzip, bzip2, xz, zip (an archive that holds one file) and zstd files, which it recognises by their first bytes, not by their names (zstd needs Python 3.14 or later, or the `zstandard` package). A compressed file it cannot decompress raises `tfbs.bedio.BedReadError`, an `OSError` that says why: corrupt or truncated data, a zip archive with no file or more than one, or zstd without a decoder.

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

The reference-data variables (`TFBS_GENOME_FASTA`, `TFBS_MOTIF_FILE`, `TFBS_FOLDX_PDB`, `TFBS_CLINVAR_VCF`, `TFBS_PROJECT_ROOT`) are read from the server's environment, so export them in the shell before starting the Inspector. `tfbs_server_info` reports the four reference files as `reference_files`, one entry per variable with `path` (the value, expanded and made absolute as the tools resolve it, or null when unset) and `exists` (whether that path exists, null when unset), so calling it in the Inspector shows whether the server sees them; `TFBS_PROJECT_ROOT` it reports as `project_root`.

The server imports PyTorch, Lightning, scikit-learn, captum, tangermeme and the genomics libraries before it answers the first message, so starting it takes about 7 to 9 seconds on a workstation (7.4 to 8.7 s measured on one Windows 10 machine with a CPU build of PyTorch) and every tool call answers at once afterwards. This is required on Windows: a native extension imported for the first time after the stdio loop has started never finishes loading there. A tool that adds a function-level import of a library with native code must add it to `_NATIVE_MODULES` in `tfbs/mcp/server.py`; a test checks this.

## Recording a session

`scripts/record_demo_session.py` (in a clone; it is not installed) starts `tfbs-mcp` over the stdio protocol as an AI app does, runs a plan of user messages, the assistant's stated plan and tool calls, and writes every call's arguments, full result and wall time to a JSON log, together with the seconds from launch to the server's first answer. The startup time quoted in the README was measured this way, and so was an earlier draft of the application note's Figure 2, with the released FLANK checkpoint (doi:10.5281/zenodo.21399578) and the bundled demo data; the published Figure 2 is a Claude Code session converted by `scripts/session_from_call_log.py` (below).

The demo directory is found with `python -c "from tfbs.mcp.cli import demo_dir; print(demo_dir())"`. Asking `tfbs.data.demo` itself does not work: it is a namespace package (it has no `__init__.py`), so its `__file__` is `None`.

```bash
python scripts/record_demo_session.py --server "$(which tfbs-mcp)" --plan plan.json --out session_log.json \
    --env TFBS_CALL_LOG=calls.jsonl --set DEMO="$(python -c "from tfbs.mcp.cli import demo_dir; print(demo_dir())")"
```

The plan is a JSON list of steps: `{"role": "user", "text": ...}`, `{"role": "llm", "text": ...}` or `{"role": "tool", "tool": ..., "arguments": {...}}`. A tool argument `{"$ref": "3.files.sequences"}` takes that value from the result of step 3. `--set KEY=VALUE` (repeatable) replaces `@KEY` by `VALUE` in every string of the plan before the references are resolved, so one plan runs on any machine, e.g. `"@DEMO/demo_peaks.bed"`. `--env KEY=VALUE` sets a variable for the server; with `TFBS_CALL_LOG` the server's own call log, which holds every call's arguments and result, is copied into the recording. A step that fails, such as a `$ref` naming a key the earlier result does not have (the log entry then lists the keys it does have) or an MCP protocol error, is logged as an error entry and the run goes on, and the log is written even when the run crashes.

The scripted demo session is `docs/demo_session_plan.json`. It needs three `--set` values: `DEMO`, the demo directory; `RELEASE`, the unpacked Zenodo archive (doi:10.5281/zenodo.21399578) or a clone of the nkx2-1-tfbs-variants repository, from which it loads `models/flank/vcnnbpnet.pt`; and `WORK`, a directory for the files the tools write. Its last call fetches PDB entry 9U19 from RCSB, so it needs internet access. Run from the repository root:

```bash
RUN=/tmp/figure2 && mkdir -p "$RUN/project" "$RUN/work"
python scripts/record_demo_session.py --server "$(which tfbs-mcp)" \
    --plan docs/demo_session_plan.json --out "$RUN/session_log.json" \
    --commit "$(git rev-parse --short HEAD)" \
    --env TFBS_PROJECT_ROOT="$RUN/project" --env TFBS_CALL_LOG="$RUN/calls.jsonl" \
    --set DEMO="$(python -c "from tfbs.mcp.cli import demo_dir; print(demo_dir())")" \
    --set RELEASE=/path/to/nkx2-1-tfbs-variants --set WORK="$RUN/work" \
    --timeout 900
```

### The smoke-test plan

`docs/all_tools_plan.json` is a plan for the same recorder that calls every one of the 33 tools at least once, in 49 steps, on the NKX2-1 data of a SLURM cluster (a ChIP-Atlas and a ReMap peak BED, the ChIP-Atlas intersect files, hg38, the JASPAR motif, the ClinVar VCF, the repaired NKX2-1 complex, FoldX and fimo on PATH); later steps take their inputs from earlier results. `docs/all_tools_runbook.md` says what to stage, gives the command with every `--set` and `--env`, lists which steps are expected to fail in which environment, and shows how to read the log; `tests/test_all_tools_plan.py` checks the plan against the live tool schemas. All 49 steps succeeded on the BIH cluster in about 50 minutes on 2026-10-13 (commit 6fbec08) and again on 2026-10-14 (040496a); the runbook has the per-phase times and how to check that the sweep's trial trained. A failed step in such a run is a failure of the server, a pipeline or the environment, never of a client's choice of arguments.

### The call log

When `TFBS_CALL_LOG` names a file, the server appends one JSON line to it after every tool call returns: `timestamp` (start of the call, UTC, milliseconds), `tool`, `arguments` as received, `seconds` (wall time), `is_error` (true when the answer has an `"error"` key or the call raised) and `result`, the tool's parsed JSON answer (`{"text": ...}` when the answer is not JSON). A result whose JSON exceeds 1 MB is stored as `{"truncated": true, "bytes": N, "head": <first 4096 characters>}`. The size is checked on the returned text before it is parsed, so a large answer is never parsed just to be cut. A call that raises is logged with `{"error": "<exception type>: <message>"}` as its result (the type name alone when the message cannot be built) and the exception reaches the client as before; an answer the hook cannot log is recorded as `{"error": "result not loggable: ..."}` and still returned unchanged. The file is append-only UTF-8 with LF line ends, and a failure to write it is a warning on stderr that never breaks the call.

### Reply size

Claude Code closes a stdio server once one message exceeds 16 MB ("wrote >16MB to stdout without a JSON-RPC message boundary"), and every model loaded with `tfbs_load_model` is lost with it; a message carries the reply text twice (`content` and `structuredContent`), escaped, so it is 2 to 4 times the reply. A `tfbs_predict` reply for 220,293 sequences in window mode, about 15.5 MB, did that on 19 October 2026. The tools with per-item results therefore keep their replies within `TFBS_MAX_INLINE_BYTES` (default 512 KiB, at most 1 MiB, so the call log keeps them whole too), in this order, each step only while the reply is still too large: arrays already saved as `.pt` files (`tfbs_analyze`, `tfbs_mutagenesis`) are left out and listed in `omitted_inline`; the per-item lists (`ids`, `scores`, `best_scores`, `mean_scores`, `best_sequences`, `predictions_before`, `predictions_after`, `delta_predictions`, `attribution_sums`, and the FASTA `ids` of `tfbs_mutagenesis`) go to a CSV whose columns carry the inline key names, one row per item, described in `results_file` (`path`, `format`, `rows`, `columns`, `sha256`) with a `summary` of each numeric column; and a reply that is still larger than the limit and than 64 KiB is written whole as JSON and named in `reply_file`. Every such reply has `"inline": false` and a `note` with the exact reading settings: the floats are written with `repr`, so the `csv` module with `float()` reads them back exactly, and so does `pandas.read_csv(path, float_precision="round_trip", keep_default_na=False, na_values={<numeric columns>: ["nan"]}, dtype={<text columns>: str})`, while pandas' defaults round some floats and turn ids such as `NA` into NaN. `tfbs_classify_metrics` writes its two curves as `roc_curve-<hash>.csv` and `pr_curve-<hash>.csv` under `results_files`. The files are named after the first 12 hex digits of their sha256 and written through a temporary file, so a later call (another checkpoint under the same model id, new random backgrounds) never changes a file that an earlier reply names. They go next to the tool's other results: `<results>/<model id>/<tool>/<digest>/`, or `output_dir` for `tfbs_analyze` and `tfbs_mutagenesis`. A directory that cannot be written gives an `{"error", "fix"}` reply. As a last resort for every tool, the tool-call hook saves any reply above 2 MiB (`REPLY_HARD_CAP_BYTES`) to `<results>/replies/<tool>-<hash>.json` and returns a short reply naming it. `tfbs_server_info` reports the limit as `max_inline_bytes`.

### A session played by a real client

`scripts/session_from_call_log.py` (in a clone; it is not installed) is the route for a session typed into a real client such as Claude Code rather than played from a plan. It turns the call log, and optionally the client's transcript, into the same JSON layout `record_demo_session.py` writes, so the same renderer draws the figure from it.

1. Register the server for the recording with the call log among its variables, at `--scope local` (this project directory only), so that sessions in other projects and subagents elsewhere do not append to the log. Use a fresh `calls.jsonl` for every session: the log is append-only, and a new directory per session keeps earlier runs out of it. `--env` takes all variables at once, so repeat the ones already registered (`claude mcp get tfbs` lists them):

   ```bash
   RUN=/path/to/session-$(date +%Y%m%d-%H%M%S) && mkdir -p "$RUN"
   claude mcp add tfbs --scope local --env TFBS_CALL_LOG="$RUN/calls.jsonl" \
       TFBS_GENOME_FASTA=/data/hg38/hg38.fa TFBS_PROJECT_ROOT=/data/tfbs -- "$(which tfbs-mcp)"
   ```

   A local registration takes precedence over a user-scope one of the same name in this directory. If you register at `--scope user` instead, remove `TFBS_CALL_LOG` afterwards (step 4), or every later Claude Code session keeps appending to the file.

2. Start Claude Code, check with `tfbs_server_info` that `call_log` is the file above, and run the session.
3. Convert. Claude Code keeps the transcript as `~/.claude/projects/<working directory with every character other than a letter, digit or - replaced by ->/<session id>.jsonl`; the newest file there is usually the session just played.

   ```bash
   python scripts/session_from_call_log.py --call-log "$RUN/calls.jsonl" --out "$RUN/session_log.json" \
       --transcript ~/.claude/projects/<project>/<session id>.jsonl \
       --commit "$(git rev-parse --short HEAD)" --client "Claude Code $(claude --version | cut -d' ' -f1)"
   ```

4. Remove the recording registration: `claude mcp remove tfbs --scope local` (or, after a user-scope registration, register it again without `TFBS_CALL_LOG`).

Without `--transcript` the steps are the tool calls of the call log. With it, the user's messages and the assistant's text are interleaved with the tool calls in timestamp order. Each `mcp__<server>__<tool>` call of the transcript is paired with the call-log entry of the same tool and equal arguments whose timestamp is the closest at or after the call's own (an entry up to 5 s earlier counts when none is later; an entry that starts after the call's result came back never does), so a call denied by the user, a call made by a subagent, or entries the server wrote in completion order shift nothing. Only a call whose arguments equal no entry falls back to the next unused entry of the same tool inside the same time window; `converted_from.matched_by_order` counts those. Entries that start outside the transcript's time span come from other sessions and are left out, counted as `calls_outside_window`; the other unmatched entries become steps of their own (`call_log_only`), and transcript calls without an entry take their result from the transcript (`transcript_only`).

The transcript fills in the result of an entry that has none (a log from before results were logged). For an entry whose result was truncated (over 1 MB), the transcript's result is used only when it parses as JSON; otherwise the step keeps the logged first 4096 characters as `result_text`, with `result_truncated: true` and `result_bytes`. Claude Code replaces any output over about 30 KB with a `<persisted-output>` "Output too large (N KB). Full output saved to: ..." note naming a local file, and that note is never used as a result, whatever the call log holds: a call with no logged result (`transcript_only`, or a log from before results were logged) then gets an empty `result_text` and `result_truncated: true`, so no local path reaches the session log. When the client's tool_result is an error but the server logged a success (the client timed out, e.g. `MCP error -32001: Request timed out`, and the server finished later), the step keeps the server's result and adds `client_saw_error: true` with the client's text as `client_error_text`. The client's own tools (Bash, Read, ...) are kept as steps with `"kind": "client"` and phase `Client`, so the figure can show that the client read a file. Subagent lines are skipped, and so is everything the client injects itself: `isMeta` and `isCompactSummary` lines, task notifications (`origin.kind` `task-notification`, written when Claude Code moves a long call to the background), and user text starting with `<command-name>`, `<command-message>`, `<local-command-stdout>`, `<local-command-caveat>`, `<task-notification>`, `<system-reminder>` or `[Request interrupted`.

A session can also take its prompts from another Claude Code session over cross-session messaging instead of from the keyboard. Such a prompt is written as an `isMeta` user line, or as a `queued_command` attachment when it arrives while the session is working, with `origin.kind` `peer`, and it is not skipped: it becomes a user step whose `text` is the prompt as sent (`origin.body`, else the text inside the `<cross-session-message>` wrapper, without the advice the client adds around it), with `"source": "peer"`, `"peer"` (the sending session's name) and `"peer_from"` (its address). These fields record that the prompt was composed outside the recorded session and sent by another session, not typed by the user. A user line stands at its own time. An attachment is stamped with the time the prompt arrived, but the model sees the prompt only when the client hands it over, so the step stands at the time of the `queue-operation` `remove` or `dequeue` line whose `commandUuid` is the attachment's `source_uuid` (or whose `deliveryId` is its `delivery_id`), with the arrival time kept as `"arrived_at"`; without such a line it stands at the arrival time. Each prompt is expected once: a second copy from the same sender with the same text, within 5 s of the time the first arrived or was handed to the model, is dropped. A `SendMessage` call (the client's own tool, through which the recorded session answers) with a non-empty `message` becomes an llm step whose `text` is that message, with `"via": "SendMessage"` and `"to"`, at the time of the call, and it is not counted among the client calls; this holds for every such call, whoever it is addressed to. When the client's tool_result for it is an error, the step also gets `"is_error": true` and the client's text as `client_error_text`, since the message may not have been delivered. A `SendMessage` call without message text stays a client step. `converted_from` counts the two as `peer_prompts` and `sendmessage_replies`.

Each tool step gets a phase: `--phases phases.json`, a JSON object mapping a tool name or an fnmatch pattern to a label, overrides the default map (Data prep: `tfbs_extract_loci`, `tfbs_prepare_*`, `tfbs_preprocess`; Training: `tfbs_config`, `tfbs_train`, `tfbs_sweep`, `tfbs_validate`, `slurm_*`; Validation: `tfbs_remap_validation`, `tfbs_chipseq_*`, `tfbs_disease_variant_validation`, `tfbs_classify_metrics`; Structure: `tfbs_fetch_structure`, `tfbs_binding_*`; Analysis: everything else). `--labels labels.json`, a JSON object mapping a 1-based index among the tfbs-mcp tool steps (`"3"`; client steps are not counted) or a tool name to a short label, writes that label as the step's `shown_args`, which the renderer prints inside the brackets after the tool name, e.g. `{"tfbs_train": "SLURM", "5": "ref", "6": "alt"}`; an index wins over a tool name, and without `--labels` no step has `shown_args`. The output's `converted_from` block holds the counts above and the client calls. The log records `seconds_to_initialize` and `tools_listed` as null, since a real client's session measures neither, and `client_server_name` is the name the client registered the server under (the `<server>` of `mcp__<server>__<tool>`), not the server's own name.

## Optional extras

The install command in the README already includes everything the server needs. This table is only for choosing a different set:

| Extra | Installs | Needed for |
|---|---|---|
| `mcp` | the MCP protocol library | the server itself |
| `genomics` | pyfaidx, tangermeme, pysam (Linux and macOS; Windows has no pysam wheel) | reading a genome FASTA (`tfbs_extract_loci`, the benchmark and validation tools); pysam reads the ClinVar VCF in `tfbs_disease_variant_validation`, which therefore needs Linux, macOS or WSL |
| `viz` | captum, tangermeme, logomaker, seaborn | attribution, mutagenesis, seqlets; tangermeme also brings its FIMO reimplementation (memelite, numba-based, shipped with tangermeme 1.x) that `tfbs_fimo` uses when the MEME Suite's `fimo` binary is not on PATH (its result says `"backend": "memelite"`, `"tangermeme"` for the `tangermeme.tools.fimo` of older releases, or `"meme"`) |
| `tracking` | wandb | `tfbs_sweep`, and Weights & Biases logging during training |
| `foldx` | pyBigWig | `tfbs_binding_scan`, which writes its result as a bigWig track and checks for pyBigWig before it scans or submits anything; not FoldX itself, see [Compute structure-based binding energy](../README.md#compute-structure-based-binding-energy) in the README |
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

`VCNNBpnet.yaml` and `RNN.yaml` carry the tuned settings from `do_train_final.sh`, with one exception below; the rest use class defaults (apart from `input_length`, see below), which for `FlexibleCNN` include four input channels for the one-hot input. All thirteen instantiate at a 24 bp window and take one-hot `(batch, 4, L)` input. `FlexibleMLP` flattens that input and sizes its first layer from `input_length`, so its template sets `input_length: 96` (4 x 24) and `tfbs_config` sets it to 4 x the window. `MultiCNN` also needs a covariate tensor, as noted in its file.

`VCNNBpnet.yaml` has a three-layer default, `dilations: [1, 1, 2]`; every other value is the study's. The study models (the NKX2-1 checkpoints and the released `vcnnbpnet.pt` files) use the five-layer stack `[1, 1, 2, 4, 8]`: every study `VCNNBpnet` in `do_train_final.sh` (all, core and flank) has it, and the template's other values are the `all_mean` ones. Trained as a classifier on a small set at batch size 64, that five-layer stack can sit at a constant output (loss ln 2, AUC 0.5): its output at initialisation varies with a std of about 1e-6 across random inputs, against about 1e-4 for three layers. On planted-motif sets of 24 bp windows (Adam, learning rate 0.001) the five-layer stack stayed constant in 3 of 10 seeds on 2,000 sequences (motif in half the positives, 10 epochs) and stayed at a loss of ln 2 in 3 of 10 seeds on 3,000 sequences (motif in every positive, 3 epochs; there its output still varied a little and its AUC was not 0.5), while the three-layer stack trained in every seed. Where the five-layer stack did train, the three-layer stack matched it: best validation loss 0.549 against 0.541 and AUC 0.749 against 0.745 on the 2,000-sequence set. On about 2,000 ChIP-seq windows the five-layer stack stayed constant for every seed and learning rate tried at batch size 64, and the three-layer stack reached a validation AUC of 0.781. Three layers have 2,340,119 parameters, five have 2,905,099; the dense head is unchanged. Changing the initialisation instead (He initialisation, with or without a rescaling for the zero-padded kernel taps) avoided the constant start but reached worse best validation losses on the 2,000-sequence set (0.627 and 0.684 on average, against 0.541 for the live five-layer runs), so the initialisation stays PyTorch's default. To train the study's stack, pass `model_params={"dilations": [1, 1, 2, 4, 8]}` to `tfbs_config`, or set `dilations` in the written config or a copy of the template. Checkpoints of the five-layer stack load as before, because the loader rebuilds the model from the `hyper_parameters` stored in the checkpoint, not from the template.

Only the template changed. The class default of `tfbs.nn.models.VCNNBpnet` is still `[1, 1, 2, 4, 8]`, so `tfbs-train fit --model=VCNNBpnet` without the template builds five layers. The `BPNet.yaml` template also uses `[1, 1, 2, 4, 8]`, and its classifier output at initialisation is nearly as flat (std 7.3e-6 to 8.6e-6 across random inputs for seeds 0 to 2); it may stall the same way, but it has not been trained on these sets and is unchanged.

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
├── bedio.py              BED reader: skips whole comment lines only
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
├── record_demo_session.py Records a stdio session log (see Recording a session)
├── chipseq_benchmark*.py Wrappers around tfbs.pipelines (likewise
├── remap_validation.py   validate.py and disease_variant_validation.py)
└── ...

docs/development.md       This page
docs/other-clients.md     Setup for AI apps other than Claude
docs/install-windows.md   The Windows differences in the installation
docs/troubleshooting.md   Symptoms, causes and fixes
docs/tools.md             All 33 tools by layer, one line each
docs/figure1.png          The task figure at the top of the README
docs/architecture.png     The layer diagram on docs/tools.md
docs/demo_session_plan.json A scripted demo session on the bundled data
docs/all_tools_plan.json  A smoke test that calls all 33 tools (see The smoke-test plan)
docs/all_tools_runbook.md How to run that plan on a SLURM cluster
tests/                    pytest suite
```

## Running tests

```bash
pip install ".[dev,mcp,viz,genomics]"
pytest
```

The `dev` extra is only pytest. The suite imports `tfbs.mcp.server`, so it needs `mcp` to collect at all, and the mutagenesis/marginalize tests exercise real tangermeme (from `viz` or `genomics`). Installing `dev` alone is not enough. Continuous integration runs on Python 3.10 to 3.13 with a CPU-only PyTorch build: first `tfbs-mcp --check`, then the suite. The tests that load the released NKX2-1 models run when `TFBS_RELEASED_MODELS` names a folder holding `flank/vcnnbpnet.pt` (the unpacked Zenodo archive's `models/`) and are skipped otherwise.
