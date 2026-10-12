"""Where a trained checkpoint lives, shared by the MCP tools and the pipelines.

tfbs_config writes a model's best checkpoint to
``<root>/saved_models/<dataset>/<scaling_method>/best_<model_type>.ckpt``,
and the NKX2-1 study's checkpoints follow the same layout under
``saved_models_final/``.  This module turns a models directory, a dataset
name and a model type into that path, so every consumer agrees on it.
Nothing heavy is imported here: the server imports it at call time.

scaling_method None means "standardize, else whichever single scaling
folder holds the model" (a classifier trained with scaling "none" lands
under ``none/``).  An explicit scaling_method names the one folder to use
and never falls back to another.

The lookup finds Lightning checkpoints, ``best_<model_type>.ckpt``, only.
An explicit path (``explicit`` here, ``--checkpoints NAME=PATH`` or
``--models NAME=PATH`` in the pipelines, ``checkpoints`` or ``nn_models``
in the MCP tools) may be either format: a PyTorch Lightning checkpoint or
an inference checkpoint such as the NKX2-1 study's released
``<models>/flank/vcnnbpnet.pt`` (Zenodo, doi:10.5281/zenodo.21399578).
tfbs.nn.loading.load_checkpoint tells the two apart by content.
"""

from __future__ import annotations

from pathlib import Path

PREFERRED_SCALING = "standardize"
# The pipelines' --scaling-method default: None keeps the fallback above.
DEFAULT_SCALING = None


def checkpoint_name(model_type: str) -> str:
    return f"best_{model_type}.ckpt"


def expected_checkpoint(models_dir: str | Path, dataset: str, model_type: str,
                        scaling_method: str | None = None) -> Path:
    """``<models_dir>/<dataset>/<scaling_method or standardize>/best_<model_type>.ckpt``."""
    folder = scaling_method or PREFERRED_SCALING
    return Path(models_dir) / dataset / folder / checkpoint_name(model_type)


def resolve_checkpoint(models_dir: str | Path, dataset: str, model_type: str,
                       scaling_method: str | None = None) -> tuple[Path | None, list[Path]]:
    """(checkpoint, candidates) for one dataset.

    The expected file wins when it exists.  Otherwise every
    ``<dataset>/*/best_<model_type>.ckpt`` is a candidate, and with
    scaling_method None a single candidate is used.  With an explicit
    scaling_method, or with none or several candidates, the checkpoint is
    None and the candidates say why.
    """
    expected = expected_checkpoint(models_dir, dataset, model_type, scaling_method)
    if expected.is_file():
        return expected, [expected]
    candidates = sorted((Path(models_dir) / dataset).glob(f"*/{checkpoint_name(model_type)}"))
    if scaling_method is None and len(candidates) == 1:
        return candidates[0], candidates
    return None, candidates


def resolve_checkpoints(models_dir: str | Path, datasets: list[str], model_type: str,
                        scaling_method: str | None = None,
                        explicit: dict[str, str] | None = None,
                        ) -> tuple[dict[str, Path], list[dict]]:
    """Resolve every dataset: ({name: path}, problems).

    ``explicit`` maps a dataset name to a checkpoint path (either format)
    and replaces the lookup under models_dir; its names are then the
    datasets.  Each problem
    is {dataset, expected, candidates} for a dataset that has no single
    checkpoint.
    """
    found: dict[str, Path] = {}
    problems: list[dict] = []
    if explicit:
        for name, path in explicit.items():
            p = Path(path).expanduser()
            if p.is_file():
                found[name] = p
            else:
                problems.append({"dataset": name, "expected": str(p), "candidates": []})
        return found, problems
    for name in datasets:
        path, candidates = resolve_checkpoint(models_dir, name, model_type, scaling_method)
        if path is None:
            problems.append({
                "dataset": name,
                "expected": str(expected_checkpoint(models_dir, name, model_type, scaling_method)),
                "candidates": [str(c) for c in candidates],
            })
        else:
            found[name] = path
    return found, problems


def parse_named_paths(items: list[str] | None) -> dict[str, str]:
    """``["name=path", ...]`` from the command line to {name: path}."""
    out: dict[str, str] = {}
    for item in items or []:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"expected NAME=PATH, got {item!r}")
        out[name] = path
    return out


def describe_problems(problems: list[dict]) -> str:
    """One line per dataset without a single checkpoint."""
    lines = []
    for p in problems:
        if len(p["candidates"]) > 1:
            lines.append(f"{p['dataset']}: {p['expected']} not found and several "
                         f"candidates match: {', '.join(p['candidates'])}")
        elif p["candidates"]:
            lines.append(f"{p['dataset']}: {p['expected']} not found; another scaling "
                         f"folder has {p['candidates'][0]}")
        else:
            lines.append(f"{p['dataset']}: {p['expected']} not found")
    return "; ".join(lines)
