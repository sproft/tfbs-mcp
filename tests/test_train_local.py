"""tfbs_train local=true: training on this machine, end to end and its error paths."""
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch

from tfbs.mcp import server


def run(coro):
    return json.loads(asyncio.run(coro))


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    server._ensure_imports()
    yield
    for key in [k for k in server._model_registry if k.startswith("lt_")]:
        server._model_registry.pop(key)


def _dataset(tmp_path: Path, labels: torch.Tensor, name: str = "peaks") -> Path:
    """tfbs_prepare_dataset output for random 24 bp windows with these labels."""
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 4, (len(labels), 24), generator=g)
    seqs = torch.nn.functional.one_hot(idx, 4).permute(0, 2, 1).float().contiguous()
    loci = tmp_path / f"{name}_loci"
    loci.mkdir()
    torch.save(seqs, loci / "sequences.pt")
    torch.save(labels, loci / "labels.pt")
    ds = run(server.tfbs_prepare_dataset(sequences_path=str(loci / "sequences.pt"),
                                         labels_path=str(loci / "labels.pt")))
    assert "error" not in ds, ds
    return Path(ds["output_dir"])


def _train(config: dict) -> dict:
    return run(server.tfbs_train(config_path=config["output_path"], local=True,
                                 local_timeout=900))


def test_local_regression_training_feeds_load_model_without_data_path(tmp_path):
    labels = torch.randn(48, 1, generator=torch.Generator().manual_seed(1)) * 2 + 3
    data = _dataset(tmp_path, labels)
    config = run(server.tfbs_config(model_type="SimpleCNN", data_path=str(data),
                                    max_epochs=1, batch_size=8))
    assert config["classify"] is False and config["scaling_method"] == "standardize"

    result = _train(config)
    assert result.get("mode") == "local", result
    assert result["exit_code"] == 0
    assert result["checkpoint_path"] == config["checkpoint_path"]
    assert result["checkpoint_exists"] is True
    assert isinstance(result["metrics"]["val_loss"], float)
    assert result["metrics"]["best_epoch"] == 0 and result["metrics"]["epochs_run"] == 1
    assert "val_acc" not in result["metrics"]
    assert Path(result["metrics_path"]).is_relative_to(Path(config["checkpoint_path"]).parent)
    assert result["command"][-2:] == ["--data.num_workers", "0"]
    log = Path(result["log_path"])
    assert log == Path(config["output_path"]).with_name("SimpleCNN_train.log") and log.is_file()

    loaded = run(server.tfbs_load_model("lt_reg", result["checkpoint_path"], "SimpleCNN",
                                        device="cpu"))
    assert Path(loaded["scaling_source"]) == data / "scaling_params.json"
    scores = run(server.tfbs_predict("lt_reg", sequences=["ACGT" * 6]))
    assert scores["inverse_transformed"] is True

    again = _train(config)
    assert again["exit_code"] == 0 and again["checkpoint_exists"] is True
    assert again["metrics_path"] != result["metrics_path"]
    ckpts = sorted(p.name for p in Path(config["checkpoint_path"]).parent.glob("*.ckpt"))
    assert ckpts == ["best_SimpleCNN.ckpt"]


def test_local_classifier_training_reports_accuracy_and_auc(tmp_path):
    labels = torch.tensor([[1.0], [0.0]] * 24)
    data = _dataset(tmp_path, labels, name="bound")
    config = run(server.tfbs_config(model_type="SimpleCNN", data_path=str(data),
                                    max_epochs=1, batch_size=8))
    assert config["classify"] is True and "note" in config

    result = _train(config)
    assert result.get("exit_code") == 0, result
    assert result["checkpoint_exists"] is True
    metrics = result["metrics"]
    assert {"val_loss", "val_acc", "val_auc"} <= set(metrics)
    assert all(isinstance(metrics[k], float) for k in ("val_loss", "val_acc", "val_auc"))


# ---------------------------------------------------------------------------
# Error paths and result fields, without starting a trainer
# ---------------------------------------------------------------------------

CONFIG = (
    "trainer:\n  callbacks:\n"
    "  - class_path: pytorch_lightning.callbacks.ModelCheckpoint\n"
    "    init_args: {dirpath: ckpts, filename: best_RNN}\n"
    "  logger:\n"
    "  - class_path: pytorch_lightning.loggers.CSVLogger\n"
    "    init_args: {save_dir: ckpts/logs, name: best_RNN}\n"
)


@pytest.fixture
def config_file(tmp_path):
    cfg = tmp_path / "cfg" / "RNN.yaml"
    cfg.parent.mkdir()
    cfg.write_text(CONFIG)
    return cfg


class FakeProc:
    """subprocess.Popen stand-in: runs ``action`` at start, or hangs until killed."""

    pid = 4242

    def __init__(self, cmd, kwargs, action=None, returncode=0, hang=False):
        self.cmd, self.kwargs = cmd, kwargs
        self.returncode, self.hang, self.killed = returncode, hang, False
        if action:
            action(cmd, kwargs)

    def wait(self, timeout=None):
        if self.killed or not self.hang:
            return -9 if self.killed else self.returncode
        time.sleep(min(timeout or 0.05, 0.05))
        raise subprocess.TimeoutExpired(self.cmd, timeout)

    def kill(self):
        self.killed = True


class Started(list):
    """The FakeProcs started, plus ``kills``: the process-group kills requested."""

    kills: list


@pytest.fixture
def popen(monkeypatch):
    """Install a FakeProc factory and fake group kills; returns (configure, started procs).

    os.killpg and taskkill are replaced, so no test signals a real process group.
    """
    procs = Started()
    procs.kills = []
    settings: dict = {}
    real_run = subprocess.run

    def fake(cmd, **kwargs):
        proc = FakeProc(cmd, kwargs, **settings)
        procs.append(proc)
        return proc

    def killpg(pid, sig):
        procs.kills.append(("killpg", pid))
        for proc in procs:
            proc.killed = True

    def run_cmd(cmd, *args, **kwargs):
        if list(cmd[:1]) != ["taskkill"]:
            return real_run(cmd, *args, **kwargs)
        procs.kills.append(("taskkill", list(cmd)))
        for proc in procs:
            proc.killed = True
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(server.subprocess, "Popen", fake)
    monkeypatch.setattr(server.os, "killpg", killpg, raising=False)
    monkeypatch.setattr(server.subprocess, "run", run_cmd)
    return settings, procs


def _metrics_csv(path: Path, text: str, age: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if age:
        stamp = time.time() - age
        os.utime(path, (stamp, stamp))
    return path


def test_local_timeout_kills_and_reports_the_log(popen, config_file):
    settings, procs = popen
    settings["hang"] = True
    result = run(server.tfbs_train(config_path=str(config_file), local=True, local_timeout=1))
    assert "local_timeout=1" in result["error"] and "local_timeout" in result["fix"]
    assert result["log_path"] == str(config_file.with_name("RNN_train.log"))
    assert procs[0].killed and procs[0].kwargs["stdin"] is subprocess.DEVNULL


def test_cancelling_the_call_kills_the_training(popen, config_file):
    settings, procs = popen
    settings["hang"] = True

    async def scenario():
        task = asyncio.create_task(server.tfbs_train(
            config_path=str(config_file), local=True, local_timeout=600))
        while not procs:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert procs[0].killed


def test_local_timeout_below_one_second_is_an_error(popen, config_file):
    _, procs = popen
    result = run(server.tfbs_train(config_path=str(config_file), local=True, local_timeout=0))
    assert "local_timeout" in result["error"] and result["fix"]
    assert procs == []


def test_local_failure_returns_the_exit_code_and_log_tail(popen, config_file):
    settings, _ = popen

    def failing(cmd, kwargs):
        kwargs["stdout"].write("Traceback: boom\n")
        kwargs["stdout"].flush()
        assert kwargs["cwd"] == str(config_file.parent)
        assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
        assert "tfbs.cli.train_cli" in cmd and "--trainer.max_epochs" in cmd

    settings.update(action=failing, returncode=3)
    result = run(server.tfbs_train(config_path=str(config_file), local=True,
                                   extra_args=["--trainer.max_epochs 2"]))
    assert result["exit_code"] == 3 and "boom" in result["log_tail"]
    assert result["error"] and result["fix"] and result["log_path"]


def test_local_success_reports_the_best_epoch_of_this_run(popen, config_file):
    settings, _ = popen
    ckpts = config_file.parent / "ckpts"
    _metrics_csv(ckpts / "logs" / "best_RNN" / "version_0" / "metrics.csv",
                 "epoch,step,val_loss\n0,5,0.01\n", age=3600)
    _metrics_csv(ckpts / "other" / "version_0" / "metrics.csv", "epoch,step,val_loss\n0,5,0.02\n")

    def ok(cmd, kwargs):
        (ckpts / "best_RNN.ckpt").write_text("x")
        _metrics_csv(ckpts / "logs" / "best_RNN" / "version_1" / "metrics.csv",
                     "epoch,step,train_loss,val_loss\n0,5,1.5,\n0,5,,0.75\n1,11,0.9,\n"
                     "1,11,,0.5\n2,17,,0.6\n")

    settings["action"] = ok
    result = run(server.tfbs_train(config_path=str(config_file), local=True))
    assert result["checkpoint_path"] == str((ckpts / "best_RNN.ckpt").resolve())
    assert result["checkpoint_exists"] is True
    assert result["metrics"] == {"val_loss": 0.5, "best_epoch": 1, "epochs_run": 3}
    assert Path(result["metrics_path"]).parent.name == "version_1"
    assert "note" not in result


def test_files_of_an_earlier_run_are_not_reported(popen, config_file):
    ckpts = config_file.parent / "ckpts"
    old = _metrics_csv(ckpts / "best_RNN.ckpt", "x", age=3600)
    _metrics_csv(ckpts / "logs" / "best_RNN" / "version_0" / "metrics.csv",
                 "epoch,step,val_loss\n0,5,0.1\n", age=3600)
    result = run(server.tfbs_train(config_path=str(config_file), local=True))
    assert result["exit_code"] == 0 and old.is_file()
    assert result["checkpoint_exists"] is False
    assert result["metrics"] is None and "metrics_path" not in result
    assert "predates this run" in result["note"] and "no metrics.csv" in result["note"]


def test_a_config_without_csv_logger_has_no_metrics_and_says_why(popen, tmp_path):
    cfg = tmp_path / "cfg" / "RNN.yaml"
    cfg.parent.mkdir()
    cfg.write_text(CONFIG.split("  logger:")[0])
    result = run(server.tfbs_train(config_path=str(cfg), local=True))
    assert result["metrics"] is None and "CSVLogger" in result["note"]


@pytest.mark.parametrize("val_labels, expected_auc", [([1.0, 1.0, 1.0], None), ([1.0, 0.0, 1.0], 0.75)])
def test_single_class_validation_reports_no_auc(popen, tmp_path, val_labels, expected_auc):
    data = tmp_path / "bound_dataset"
    (data / "val").mkdir(parents=True)
    torch.save(torch.tensor(val_labels).unsqueeze(1), data / "val" / "labels.pt")
    cfg = tmp_path / "cfg" / "RNN.yaml"
    cfg.parent.mkdir()
    cfg.write_text(CONFIG + f"data:\n  data_path: {data.as_posix()}\n")
    settings, _ = popen
    settings["action"] = lambda cmd, kwargs: _metrics_csv(
        cfg.parent / "ckpts" / "logs" / "best_RNN" / "version_0" / "metrics.csv",
        f"epoch,step,val_loss,val_acc,val_auc\n0,5,0.7,1.0,{expected_auc or 0.0}\n")
    result = run(server.tfbs_train(config_path=str(cfg), local=True))
    assert result["metrics"]["val_auc"] == expected_auc
    if expected_auc is None:
        assert "validation split contains one class" in result["note"]
    else:
        assert "note" not in result


def test_extra_args_tokens():
    assert server._argv_tokens("--trainer.max_epochs 3", posix=True) == (
        ["--trainer.max_epochs", "3"], None)
    assert server._argv_tokens('--x "a b"', posix=True) == (["--x", "a b"], None)
    assert server._argv_tokens('--x "a b" \'c\'', posix=False) == (["--x", "a b", "c"], None)
    windows = r"--p='a b' --ckpt C:\data\x.ckpt"
    assert server._argv_tokens(windows, posix=False) == (
        ["--p=a b", "--ckpt", r"C:\data\x.ckpt"], None)
    assert server._argv_tokens(windows, posix=True) == (["--p=a b", "--ckpt", "C:datax.ckpt"], None)
    windows_path = "--data.data_path=C:\\data\\peaks_dataset"
    assert server._argv_tokens(windows_path, posix=True) == ([windows_path], None)
    tokens, err = server._argv_tokens('--x "unclosed', posix=True)
    assert tokens == [] and "unclosed" in json.loads(err)["error"]
    assert json.loads(server._argv_tokens("--x 'unclosed", posix=False)[1])["fix"]


def test_unbalanced_extra_args_are_an_error_in_every_mode(popen, config_file):
    _, procs = popen
    for local in (True, False):
        result = run(server.tfbs_train(config_path=str(config_file), local=local,
                                       extra_args=['--trainer.max_epochs "3']))
        assert "--trainer.max_epochs" in result["error"] and result["fix"]
    assert procs == []


def test_tail_file_survives_bytes_that_are_not_utf8(tmp_path):
    log = tmp_path / "train.log"
    log.write_bytes(b"epoch 1\n\xff\xfe loss\nend\n")
    tail = server._tail_file(str(log), 2)
    assert tail.endswith("end\n") and "loss" in tail


def test_without_sbatch_the_fix_offers_local_mode(monkeypatch, config_file):
    monkeypatch.setattr(server.shutil, "which", lambda *a, **k: None)
    result = run(server.tfbs_train(config_path=str(config_file)))
    assert "sbatch" in result["error"] and "local=true" in result["fix"]


def test_local_mode_needs_a_config(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n")
    result = run(server.tfbs_train(script=str(script), local=True))
    assert "config" in result["error"] and "config_path" in result["fix"]


def test_train_description_states_local_mode():
    doc = " ".join(server.tfbs_train.__doc__.split())
    assert "local=true" in doc and "blocks until training finishes" in doc
    assert "small datasets and laptops" in doc and "one or more argv tokens" in doc
    assert "The same tokenization applies in SLURM config mode" in doc
    assert "best_epoch" in doc and "epochs_run" in doc and "single class" in doc


@pytest.fixture
def group_kill(popen):
    return popen[1].kills


def _expected_group_kill() -> tuple:
    if os.name == "nt":
        return ("taskkill", ["taskkill", "/T", "/F", "/PID", "4242"])
    return ("killpg", 4242)


def _assert_started_in_own_group(proc) -> None:
    if os.name == "nt":
        assert proc.kwargs["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert proc.kwargs["start_new_session"] is True


def test_timeout_kills_the_whole_process_group(popen, group_kill, config_file):
    settings, procs = popen
    settings["hang"] = True
    result = run(server.tfbs_train(config_path=str(config_file), local=True, local_timeout=1))
    assert "local_timeout=1" in result["error"]
    _assert_started_in_own_group(procs[0])
    assert group_kill == [_expected_group_kill()]


def test_cancellation_kills_the_whole_process_group(popen, group_kill, config_file):
    settings, procs = popen
    settings["hang"] = True

    async def scenario():
        task = asyncio.create_task(server.tfbs_train(
            config_path=str(config_file), local=True, local_timeout=600))
        while not procs:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    _assert_started_in_own_group(procs[0])
    assert group_kill[0] == _expected_group_kill()


def test_group_kill_on_each_platform_falls_back_to_kill(monkeypatch):
    killed: list = []
    proc = FakeProc(["x"], {})
    monkeypatch.setattr(server.os, "killpg", lambda pid, sig: killed.append(("killpg", pid)),
                        raising=False)
    server._kill_process_group(proc, posix=True)
    assert killed == [("killpg", 4242)] and not proc.killed

    def gone(pid, sig):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(server.os, "killpg", gone, raising=False)
    server._kill_process_group(proc, posix=True)
    assert proc.killed

    calls: list = []

    def taskkill(code):
        def fake(cmd, *a, **k):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, code, b"", b"")
        return fake

    proc = FakeProc(["x"], {})
    monkeypatch.setattr(server.subprocess, "run", taskkill(0))
    server._kill_process_group(proc, posix=False)
    assert calls == [["taskkill", "/T", "/F", "/PID", "4242"]] and not proc.killed
    monkeypatch.setattr(server.subprocess, "run", taskkill(128))
    server._kill_process_group(proc, posix=False)
    assert proc.killed

    assert server._new_process_group(posix=True) == {"start_new_session": True}
    flags = server._new_process_group(posix=False)["creationflags"]
    assert flags == getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)


def test_quoted_quotes_tokenize_the_same_on_both_platforms():
    arg = "--x \"'quoted'\" --y '\"a b\"'"
    expected = (["--x", "'quoted'", "--y", '"a b"'], None)
    assert server._argv_tokens(arg, posix=True) == expected
    assert server._argv_tokens(arg, posix=False) == expected


OLD_ROWS = "epoch,step,val_loss\n0,5,0.01\n1,11,0.02\n"


def _fixed_version_run(popen, config_file, write) -> dict:
    settings, _ = popen
    csv_path = config_file.parent / "ckpts" / "logs" / "best_RNN" / "version_0" / "metrics.csv"
    _metrics_csv(csv_path, OLD_ROWS, age=3600)
    settings["action"] = lambda cmd, kwargs: write(csv_path)
    return run(server.tfbs_train(config_path=str(config_file), local=True))


def test_fixed_version_appended_rows_of_an_earlier_run_are_skipped(popen, config_file):
    def append(path):
        with open(path, "a", newline="") as f:
            f.write("0,5,0.9\n1,11,0.7\n2,17,0.8\n")

    result = _fixed_version_run(popen, config_file, append)
    assert result["metrics"] == {"val_loss": 0.7, "best_epoch": 1, "epochs_run": 3}


def test_fixed_version_rewritten_with_a_wider_header_skips_the_old_rows(popen, config_file):
    def rewrite(path):
        path.write_text("epoch,step,train_loss,val_loss\n0,5,,0.01\n1,11,,0.02\n"
                        "0,5,1.5,\n0,5,,0.9\n")

    result = _fixed_version_run(popen, config_file, rewrite)
    assert result["metrics"] == {"val_loss": 0.9, "best_epoch": 0, "epochs_run": 1}


@pytest.mark.parametrize("rows", [OLD_ROWS, OLD_ROWS + "2,17,0.005\n"])
def test_a_rerun_that_logs_the_same_rows_again_is_read_whole(popen, config_file, rows):
    result = _fixed_version_run(popen, config_file, lambda path: path.write_text(rows))
    assert result["metrics"]["val_loss"] == min(0.01, 0.005 if "2,17" in rows else 1.0)
    assert result["metrics"]["epochs_run"] == rows.count("\n") - 1


def test_train_description_states_the_fixed_version_and_group_kill():
    doc = " ".join(server.tfbs_train.__doc__.split())
    assert "A CSVLogger with a fixed version reuses its metrics.csv" in doc
    assert "rows the file held when the run started are skipped" in doc
    assert "that whole group (the process tree, on Windows) is killed" in doc
    assert "on Windows the quotes around each token are then removed" not in doc
