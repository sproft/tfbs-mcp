"""tfbs_train's generated SLURM job and slurm_logs' search for the log files."""
import asyncio
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tfbs.mcp import server


def run(coro):
    return json.loads(asyncio.run(coro))


# ---------------------------------------------------------------------------
# tfbs_train, SLURM config mode
# ---------------------------------------------------------------------------

class Sbatch:
    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(server.shutil, "which",
                            lambda name, *a, **k: f"/usr/bin/{name}" if name == "sbatch" else None)
        monkeypatch.setattr(server.subprocess, "run", self._run)

    def _run(self, cmd, *a, **k):
        self.calls.append({"cmd": list(cmd), "cwd": k.get("cwd")})
        return subprocess.CompletedProcess(cmd, 0, "Submitted batch job 77\n", "")


def _config(tmp_path, num_workers=24, dirpath="ckpts", filename="best_VCNNBpnet"):
    config = {
        "data": {"data_path": "data/x_dataset", "batch_size": 64},
        "trainer": {"callbacks": [{
            "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
            "init_args": {"dirpath": dirpath, "filename": filename, "monitor": "val_loss"},
        }]},
    }
    if num_workers is not None:
        config["data"]["num_workers"] = num_workers
    path = tmp_path / "cfg" / "VCNNBpnet.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_config_mode_returns_the_config_and_checkpoint_paths(tmp_path, monkeypatch):
    rec = Sbatch(monkeypatch)
    config = _config(tmp_path)
    result = run(server.tfbs_train(config_path=str(config)))
    assert result["job_id"] == "77", result
    assert result["config_path"] == str(config.resolve())
    assert result["checkpoint_path"] == str((config.parent / "ckpts" / "best_VCNNBpnet.ckpt").resolve())
    assert rec.calls[-1]["cwd"] == str(config.parent.resolve())


def test_config_mode_checkpoint_path_matches_local_mode(tmp_path):
    config = _config(tmp_path, dirpath=str(tmp_path / "abs" / "models"), filename="best")
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    target = server._checkpoint_target(loaded, config.parent)
    assert target == (tmp_path / "abs" / "models" / "best.ckpt").resolve()


def test_config_without_a_checkpoint_callback_gives_a_null_checkpoint_path(tmp_path, monkeypatch):
    Sbatch(monkeypatch)
    config = tmp_path / "plain.yaml"
    config.write_text("model: {}\n", encoding="utf-8")
    result = run(server.tfbs_train(config_path=str(config)))
    assert result["checkpoint_path"] is None and result["config_path"] == str(config.resolve())


def test_cpus_default_to_the_configs_num_workers_plus_the_main_process(tmp_path, monkeypatch):
    Sbatch(monkeypatch)
    result = run(server.tfbs_train(config_path=str(_config(tmp_path, num_workers=24))))
    script = result["generated_script"]
    assert "#SBATCH --cpus-per-task=25\n" in script
    assert "--mem" not in script
    assert result["cpus_per_task"] == 25 and result["mem"] is None


@pytest.mark.parametrize("workers, expected", [(0, 1), (None, 1)])
def test_cpus_are_at_least_one(tmp_path, monkeypatch, workers, expected):
    Sbatch(monkeypatch)
    result = run(server.tfbs_train(config_path=str(_config(tmp_path, num_workers=workers))))
    assert f"#SBATCH --cpus-per-task={expected}\n" in result["generated_script"]


def test_extra_args_num_workers_sets_the_default_cpus(tmp_path, monkeypatch):
    Sbatch(monkeypatch)
    config = str(_config(tmp_path, num_workers=24))
    result = run(server.tfbs_train(config_path=config, extra_args=["--data.num_workers 4"]))
    assert "#SBATCH --cpus-per-task=5\n" in result["generated_script"]
    result = run(server.tfbs_train(config_path=config, extra_args=["--data.num_workers=6"]))
    assert "#SBATCH --cpus-per-task=7\n" in result["generated_script"]


def test_explicit_cpus_and_mem_are_written_to_the_job(tmp_path, monkeypatch):
    rec = Sbatch(monkeypatch)
    result = run(server.tfbs_train(config_path=str(_config(tmp_path)), cpus_per_task=8,
                                   mem="64g"))
    script = result["generated_script"]
    assert "#SBATCH --cpus-per-task=8\n#SBATCH --mem=64G\n" in script
    assert result["cpus_per_task"] == 8 and result["mem"] == "64G"
    written = Path(rec.calls[-1]["cmd"][-1]).read_bytes()
    assert b"#SBATCH --mem=64G\n" in written and b"\r\n" not in written


@pytest.mark.parametrize("kwargs, words", [({"cpus_per_task": 0}, "cpus_per_task"),
                                           ({"mem": "lots"}, "mem"),
                                           ({"mem": "0"}, "mem must be at least 1"),
                                           ({"mem": "0G"}, "mem must be at least 1"),
                                           ({"mem": "64G\n#SBATCH --x"}, "mem")])
def test_bad_cpus_or_mem_are_errors_before_submitting(tmp_path, monkeypatch, kwargs, words):
    rec = Sbatch(monkeypatch)
    result = run(server.tfbs_train(config_path=str(_config(tmp_path)), **kwargs))
    assert words in result["error"] and result["fix"]
    assert rec.calls == []


def test_an_unreadable_config_is_an_error_in_slurm_mode(tmp_path, monkeypatch):
    rec = Sbatch(monkeypatch)
    config = tmp_path / "broken.yaml"
    config.write_text("trainer: [unclosed\n", encoding="utf-8")
    result = run(server.tfbs_train(config_path=str(config)))
    assert "Could not read the config" in result["error"]
    assert rec.calls == []


def test_train_docstring_documents_the_new_parameters():
    doc = " ".join(server.tfbs_train.__doc__.split())
    assert "cpus_per_task:" in doc and "data.num_workers" in doc
    assert "mem:" in doc and "no --mem line" in doc
    assert "checkpoint_path" in doc and "config_path" in doc


# ---------------------------------------------------------------------------
# slurm_logs
# ---------------------------------------------------------------------------

class FakeScheduler:
    """sacct and scontrol answering with prepared text."""

    def __init__(self, monkeypatch, sacct=None, scontrol=None, old_sacct=False):
        self.sacct, self.scontrol, self.old_sacct = sacct, scontrol, old_sacct
        self.calls = []
        monkeypatch.setattr(server.subprocess, "run", self._run)

    def _run(self, cmd, *a, **k):
        self.calls.append(list(cmd))
        if cmd[0] == "sacct":
            if self.sacct is None:
                raise FileNotFoundError("sacct")
            fmt = next(c for c in cmd if c.startswith("--format="))
            if self.old_sacct and "StdOut" in fmt:
                return subprocess.CompletedProcess(cmd, 1, "", "sacct: error: Invalid field requested: \"StdOut\"\n")
            return subprocess.CompletedProcess(cmd, 0, self.sacct, "")
        if cmd[0] == "scontrol":
            if self.scontrol is None:
                raise FileNotFoundError("scontrol")
            return subprocess.CompletedProcess(cmd, 0, self.scontrol, "")
        raise AssertionError(cmd)


def _job_dir(tmp_path):
    work = tmp_path / "cfg"
    work.mkdir()
    (work / "tfbs_train_VCNNBpnet_123.out").write_text("epoch 1\nepoch 2\n", encoding="utf-8")
    (work / "tfbs_train_VCNNBpnet_123.err").write_text("warning\nTraceback: boom\n", encoding="utf-8")
    return work


def test_relative_sacct_paths_are_joined_to_workdir_and_expanded(tmp_path, monkeypatch):
    work = _job_dir(tmp_path)
    FakeScheduler(monkeypatch, sacct=(
        f"123|tfbs_train_VCNNBpnet|{work}|%x_%j.out|%x_%j.err\n"
        f"123.batch|batch|{work}||\n"))
    result = run(server.slurm_logs(job_id="123", log_type="both"))
    assert result["log_paths"] == {"stdout": str(work / "tfbs_train_VCNNBpnet_123.out"),
                                   "stderr": str(work / "tfbs_train_VCNNBpnet_123.err")}
    assert result["log_sources"] == {"stdout": "sacct", "stderr": "sacct"}
    assert "epoch 2" in result["stdout"] and "Traceback: boom" in result["stderr"]


def test_expand_slurm_pattern():
    assert server._expand_slurm_pattern("%x_%j.out", "9", "job") == "job_9.out"
    assert server._expand_slurm_pattern("100%%_%j", "9", "job") == "100%_9"
    assert server._expand_slurm_pattern("%A_%a.out", "9", "job") == "%A_%a.out"


def test_a_missing_sacct_file_falls_back_to_scontrol(tmp_path, monkeypatch):
    work = _job_dir(tmp_path)
    elsewhere = tmp_path / "logs" / "job.err"
    elsewhere.parent.mkdir()
    elsewhere.write_text("from scontrol\n", encoding="utf-8")
    fake = FakeScheduler(monkeypatch, sacct=f"123|tfbs_train_VCNNBpnet|{work}|gone.out|gone.err\n",
                         scontrol=(f"JobId=123 JobName=tfbs_train_VCNNBpnet\n"
                                   f"   WorkDir={work}\n   StdErr={elsewhere}\n"
                                   f"   StdIn=/dev/null\n   StdOut={elsewhere}\n"))
    result = run(server.slurm_logs(job_id="123"))
    assert result["log_paths"]["stderr"] == str(elsewhere)
    assert result["log_sources"]["stderr"] == "scontrol"
    assert "from scontrol" in result["stderr"]
    assert ["scontrol", "show", "job", "123"] in fake.calls


def test_without_any_reported_file_the_default_name_in_workdir_is_read(tmp_path, monkeypatch):
    work = _job_dir(tmp_path)
    FakeScheduler(monkeypatch, sacct=f"123|tfbs_train_VCNNBpnet|{work}\n", old_sacct=True,
                  scontrol="")
    result = run(server.slurm_logs(job_id="123", log_type="both"))
    assert result["log_sources"] == {"stdout": "default name", "stderr": "default name"}
    assert "epoch 2" in result["stdout"] and "Traceback: boom" in result["stderr"]


def test_scontrol_alone_is_enough_without_sacct(tmp_path, monkeypatch):
    work = _job_dir(tmp_path)
    FakeScheduler(monkeypatch, sacct=None,
                  scontrol=f"JobId=123 JobName=tfbs_train_VCNNBpnet\n   WorkDir={work}\n"
                           f"   StdOut={work}/tfbs_train_VCNNBpnet_123.out\n")
    result = run(server.slurm_logs(job_id="123", log_type="stdout"))
    assert result["log_sources"]["stdout"] == "scontrol" and "epoch 2" in result["stdout"]


def test_a_log_found_nowhere_is_reported_with_its_best_candidate(tmp_path, monkeypatch):
    FakeScheduler(monkeypatch, sacct=f"123|job|{tmp_path}|out_%j.txt|err_%j.txt\n", scontrol="")
    result = run(server.slurm_logs(job_id="123"))
    assert result["log_paths"]["stderr"] == str(tmp_path / "err_123.txt")
    assert result["stderr"].startswith("[File not found:")


def test_neither_sacct_nor_scontrol_is_an_error(monkeypatch):
    FakeScheduler(monkeypatch, sacct=None, scontrol=None)
    result = run(server.slurm_logs(job_id="123"))
    assert "sacct nor scontrol" in result["error"] and result["fix"]


def test_unknown_log_type_is_an_error(monkeypatch):
    FakeScheduler(monkeypatch, sacct="", scontrol="")
    assert "log_type" in run(server.slurm_logs(job_id="1", log_type="all"))["error"]


def test_a_relative_log_path_without_a_workdir_is_not_read_from_the_servers_cwd(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "job.out").write_text("unrelated file in the server's cwd\n", encoding="utf-8")
    FakeScheduler(monkeypatch, sacct="123|tfbs_train_x||job.out|job.err\n",
                  scontrol="JobId=123 JobName=tfbs_train_x\n   StdOut=job.out\n")
    result = run(server.slurm_logs(job_id="123", log_type="stdout"))
    assert result["log_paths"]["stdout"] is None
    assert "unrelated" not in result["stdout"]


def test_slurm_log_path_skips_a_relative_path_without_workdir(tmp_path):
    assert server._slurm_log_path("%x_%j.out", None, "9", "job") is None
    assert server._slurm_log_path("%x_%j.out", "/w", "9", "job") == Path("/w") / "job_9.out"
    absolute = (tmp_path / "%j.out").as_posix()
    assert server._slurm_log_path(absolute, None, "9", "job") == tmp_path / "9.out"
