"""The server must answer tools that need native extensions over real stdio.

On Windows, loading a native extension (numpy, scipy, torch) for the first
time after the stdio event loop has started never returns, so a server that
imported them lazily on the first tool call hung for every real client.  These
tests drive the real server as a subprocess over the MCP stdio protocol, which
no in-process test can stand in for.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "tfbs" / "data" / "demo"
INIT_TIMEOUT = 300  # startup imports torch, Lightning, captum and tangermeme
CALL_TIMEOUT = 180  # a hang never returns; a working call takes seconds


async def _call_over_stdio(calls: list[tuple[str, dict]], project_root: Path) -> list[dict]:
    env = {
        **get_default_environment(),
        "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])),
        "TFBS_PROJECT_ROOT": str(project_root),
    }
    # tfbs_fimo must take its fallback: no MEME Suite fimo on the server's PATH.
    path = env.get("PATH", os.environ.get("PATH", ""))
    env["PATH"] = os.pathsep.join(d for d in path.split(os.pathsep)
                                  if d and shutil.which("fimo", path=d) is None)
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "tfbs.mcp.cli"], env=env, cwd=str(REPO))
    results = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), INIT_TIMEOUT)
            for name, arguments in calls:
                result = await asyncio.wait_for(session.call_tool(name, arguments), CALL_TIMEOUT)
                text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
                results.append(json.loads(text))
    return results


def _lightningcli_checkpoint(path: Path) -> Path:
    """A SimpleCNN checkpoint as LightningCLI writes it (class path and instantiator)."""
    import pytorch_lightning as pl
    import torch

    from tfbs.nn import models

    model = models.SimpleCNN(input_length=24)
    hparams = {**model.hparams, "_class_path": "tfbs.nn.models.SimpleCNN",
               "_instantiator": "pytorch_lightning.cli.instantiate_module"}
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": hparams,
                "pytorch-lightning_version": pl.__version__}, path)
    return path


def test_native_extension_tools_answer_over_stdio(tmp_path):
    """scikit-learn, torch, pyfaidx, Lightning checkpoint loading and the numba FIMO after the loop started."""
    ckpt = _lightningcli_checkpoint(tmp_path / "best_SimpleCNN.ckpt")
    out = asyncio.run(_call_over_stdio([
        ("tfbs_classify_metrics", {"positive_scores": [0.9, 0.8], "negative_scores": [0.2, 0.1]}),
        ("tfbs_preprocess", {"sequences": ["ACGTACGTACGTACGT"],
                             "output_path": str(tmp_path / "pre")}),
        ("tfbs_extract_loci", {"bed_path": str(DEMO / "demo_peaks.bed"),
                               "genome_fasta": str(DEMO / "demo_genome.fa"),
                               "output_dir": str(tmp_path / "loci"), "in_window": 24}),
        ("tfbs_server_info", {}),
        ("tfbs_load_model", {"model_id": "m", "checkpoint_path": str(ckpt),
                             "model_type": "SimpleCNN", "device": "cpu"}),
        ("tfbs_predict", {"model_id": "m", "sequences": ["ACGTACGTACGTACGTACGTACGT"]}),
        ("tfbs_fimo", {"sequences": ["CTCACTTGACCTT"], "motif_file": str(DEMO / "MA1994.1.meme"),
                       "output_dir": str(tmp_path / "fimo")}),
    ], tmp_path))
    metrics, pre, loci, info, loaded, scores, fimo = out
    assert metrics["roc_auc"] == 1.0
    assert pre["shape"] == [1, 4, 16]
    assert "error" not in loci, loci
    assert loci["shape"][1:] == [4, 24] and loci["num_peaks_extracted"] > 0
    assert Path(loci["sequences_path"]).is_file()
    assert info["torch_version"] and "cuda_available" in info
    assert "torch_loaded" not in info
    assert loaded["input_length"] == 24, loaded
    assert len(scores["scores"]) == 1
    assert fimo["backend"] in ("memelite", "tangermeme"), fimo
    assert fimo["total_hits"] == 1 and fimo["hits"][0]["matched_sequence"] == "CTCACTTGACCTT"
