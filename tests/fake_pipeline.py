"""A stand-in for the pipeline processes the validation tools start.

The direct-mode tools start a pipeline with server._spawn_pipeline and read
it with communicate() in a worker thread.  spawn_via(run) replaces that
process with one whose communicate() calls ``run(cmd, timeout=..., **kwargs)``,
a subprocess.run look-alike returning a CompletedProcess (or raising
TimeoutExpired), so a test fakes pipeline runs and sbatch calls with one
function.
"""
from __future__ import annotations

from tfbs.mcp import server


class FakePipelineProcess:
    pid = None  # nothing real to kill

    def __init__(self, run, cmd, kwargs):
        self._run = run
        self.args = cmd
        self.kwargs = kwargs
        self.returncode = None
        self.killed = False

    def communicate(self, timeout=None):
        if self.killed:
            return "", ""
        done = self._run(self.args, timeout=timeout, **self.kwargs)
        self.returncode = done.returncode
        return done.stdout, done.stderr

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


def spawn_via(run):
    """A server._spawn_pipeline replacement that runs ``run`` instead of a process."""
    def spawn(cmd, cwd):
        return FakePipelineProcess(run, list(cmd), server._pipeline_popen_kwargs(cwd))
    return spawn


def fake_pipelines(monkeypatch, run) -> None:
    monkeypatch.setattr(server, "_spawn_pipeline", spawn_via(run))
