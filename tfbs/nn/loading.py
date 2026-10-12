"""Load a trained model from either checkpoint format, for the server and the pipelines.

Two formats are recognised by their content, not their file name:

* An inference checkpoint is a dict saved with ``torch.save`` that holds
  ``arch`` (a class name in ``tfbs.nn.models``), ``hyper_parameters``
  (its constructor arguments) and ``state_dict``, and optionally
  ``scaling`` ({scaling_method, mean, std}), ``library`` and
  ``metadata``.  The NKX2-1 study's released ``vcnnbpnet.pt`` files
  (Zenodo, doi:10.5281/zenodo.21399578; one each for the core, flank and
  all libraries) are of this kind.  It is read with weights_only=True,
  rebuilt from its hyper_parameters and loaded with strict=True.
* Anything else is loaded as a PyTorch Lightning checkpoint through
  ``<class>.load_from_checkpoint``, except two kinds of file that are
  neither: a bare state_dict (``torch.save(model.state_dict())``, a dict
  of tensors only) and a file that is no PyTorch checkpoint at all (one
  torch.load cannot read and that is not a zip archive, such as an empty
  or a truncated file).  Both are a ValueError before anything is built.  The class is model_type, or the
  class path that LightningCLI records in hyper_parameters["_class_path"]
  (e.g. ``classes.nn.models.VCNNBpnet``); the module named there is
  imported first, because Lightning refuses a class path whose module is
  not imported.

``load_checkpoint`` never moves the scores to label units: an inference
checkpoint's ``scaling`` is returned in the info for the caller to use.
The module itself imports torch and the model classes only when called
(its package, tfbs.nn, imports them anyway), and it prints nothing, so the
MCP server can use it inside its stdio process.
"""

from __future__ import annotations

import gc
import importlib
import os
import re
import sys
import zipfile
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CheckpointInfo",
    "UnimportableClassError",
    "checkpoint_problem",
    "is_bare_state_dict",
    "is_inference_payload",
    "load_checkpoint",
    "model_class",
    "model_class_names",
    "read_payload",
]

_CLASS_PATH_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_UNREAD = object()


@dataclass
class CheckpointInfo:
    """What a checkpoint says about the model it holds.

    format is "inference" or "lightning".  scaling, library and metadata
    are set for an inference checkpoint only (scaling None when the file
    has no scaling block), class_path for a Lightning checkpoint only
    (None when it records none).
    """

    format: str
    arch: str
    hyper_parameters: dict = field(default_factory=dict)
    scaling: dict | None = None
    library: Any = None
    metadata: Any = None
    class_path: str | None = None


class UnimportableClassError(ImportError):
    """A Lightning checkpoint names a class whose module cannot be imported."""

    def __init__(self, class_path: str):
        self.class_path = class_path
        self.module = class_path.rpartition(".")[0]
        super().__init__(f"The checkpoint was saved with the class {class_path}, and its "
                         f"module {self.module} cannot be imported.")


def read_payload(path) -> Any:
    """The object torch.load(weights_only=True) reads from path, or None if it cannot."""
    import torch

    for kwargs in ({"mmap": True}, {}):
        try:
            return torch.load(os.fspath(path), map_location="cpu", weights_only=True, **kwargs)
        except Exception:
            continue
    return None


def is_inference_payload(payload: Any) -> bool:
    """An inference checkpoint: a dict with arch, hyper_parameters and state_dict."""
    return (isinstance(payload, dict) and isinstance(payload.get("arch"), str)
            and isinstance(payload.get("hyper_parameters"), dict)
            and isinstance(payload.get("state_dict"), dict))


def is_bare_state_dict(payload: Any) -> bool:
    """A dict of tensors only, as torch.save(model.state_dict()) writes it.

    Such a file holds the weights but neither the architecture nor its
    constructor arguments, so no model can be rebuilt from it.
    """
    if not isinstance(payload, dict) or not payload or "state_dict" in payload:
        return False
    import torch

    return all(isinstance(value, torch.Tensor) for value in payload.values())


def checkpoint_problem(path, payload: Any) -> str | None:
    """Why the file at path is no checkpoint load_checkpoint can use, or None.

    payload is what read_payload returned for it.  A bare state_dict and
    an existing file that is no PyTorch checkpoint (torch.load cannot
    read it and it is not a zip archive) are named; anything else,
    including a missing file, is left to the loaders.
    """
    if is_bare_state_dict(payload):
        return (f"{os.fspath(path)} is a bare state_dict (a dict of tensors only), which "
                "records neither the architecture nor its hyper_parameters; save "
                "{arch, hyper_parameters, state_dict} with torch.save or use a "
                "Lightning .ckpt.")
    if payload is None and os.path.isfile(os.fspath(path)):
        try:
            archive = zipfile.is_zipfile(os.fspath(path))
        except OSError:
            archive = False
        if not archive:
            return (f"{os.fspath(path)} is not a PyTorch checkpoint: torch.load cannot read "
                    "it and it is not a zip archive (an empty, truncated or unrelated "
                    "file); pass a Lightning .ckpt or an inference checkpoint such as "
                    "a released vcnnbpnet.pt.")
    return None


def model_class_names() -> list[str]:
    """The model classes of tfbs.nn.models a model_type or an arch may name.

    A class qualifies when it is a torch.nn.Module defined in
    tfbs.nn.models under its own name; BaseModel, private names and
    classes imported from elsewhere (equirc's layers, or their stand-in
    when equirc is missing) do not.  The server lists the same classes
    as the valid model types.
    """
    import torch

    import tfbs.nn.models as models

    names = []
    for name in dir(models):
        cls = getattr(models, name, None)
        if (isinstance(cls, type) and issubclass(cls, torch.nn.Module)
                and name != "BaseModel" and not name.startswith("_")
                and cls.__module__ == models.__name__ and cls.__name__ == name):
            names.append(name)
    return sorted(names)


def model_class(name: str):
    """The class tfbs.nn.models.<name>; ValueError naming the choices otherwise."""
    import tfbs.nn.models as models

    known = model_class_names()
    if not isinstance(name, str) or name not in known:
        raise ValueError(f"{name!r} is not a model class in tfbs.nn.models "
                         f"(known: {', '.join(known)}).")
    return getattr(models, name)


def recorded_class_path(payload: Any) -> str | None:
    """hyper_parameters["_class_path"] of a LightningCLI checkpoint, or None."""
    hparams = payload.get("hyper_parameters") if isinstance(payload, dict) else None
    value = hparams.get("_class_path") if isinstance(hparams, dict) else None
    return value if isinstance(value, str) and "." in value else None


def import_class_module(class_path: str) -> bool:
    """Import the module that defines class_path; False if it cannot be imported."""
    module = class_path.rpartition(".")[0]
    if not module:
        return False
    if module in sys.modules:
        return True
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def class_path_in(message: str, class_name: str) -> str | None:
    """A class path ending in class_name, from a module not yet imported, in a message."""
    for match in _CLASS_PATH_RE.findall(message):
        module, _, name = match.rpartition(".")
        if name == class_name and module not in sys.modules:
            return match
    return None


def _finish(model, map_location):
    model.eval()
    if map_location is not None and hasattr(model, "to"):
        model = model.to(map_location)
    return model


def _load_inference(payload: dict, model_type: str | None, map_location, overrides: dict):
    arch = payload["arch"]
    if model_type is not None and arch != model_type:
        raise ValueError(f"The checkpoint holds a {arch} model, but model_type is "
                         f"{model_type}; pass model_type {arch!r}.")
    cls = model_class(arch)
    hyper_parameters = {**payload["hyper_parameters"], **overrides}
    model = cls(**hyper_parameters)
    model.load_state_dict(payload["state_dict"], strict=True)
    scaling = payload.get("scaling")
    info = CheckpointInfo(
        format="inference", arch=arch, hyper_parameters=hyper_parameters,
        scaling=scaling if isinstance(scaling, dict) else None,
        library=payload.get("library"), metadata=payload.get("metadata"),
    )
    del payload
    # Building the model ran Lightning's save_hyperparameters, whose
    # reference cycle keeps the frames on the stack alive with the locals
    # they return with, and the frames above this one (a caller's too) may
    # hold the memory-mapped payload.  Collecting the cycle now frees them,
    # so the checkpoint file does not stay mapped (on Windows a mapped file
    # cannot be overwritten).
    gc.collect()
    return _finish(model, map_location), info


def _lightning_record(payload: Any) -> dict:
    """The class path and hyper_parameters of a Lightning checkpoint, without its tensors."""
    hparams = payload.get("hyper_parameters") if isinstance(payload, dict) else None
    return {"class_path": recorded_class_path(payload),
            "hyper_parameters": dict(hparams) if isinstance(hparams, dict) else {}}


def _load_lightning(path, record: dict, model_type: str | None, map_location, overrides: dict):
    class_path = record["class_path"]
    if model_type is not None:
        cls = model_class(model_type)
    elif class_path:
        if not import_class_module(class_path):
            raise UnimportableClassError(class_path)
        module, _, name = class_path.rpartition(".")
        cls = getattr(sys.modules[module], name, None)
        if not isinstance(cls, type):
            raise ValueError(f"The checkpoint names the class {class_path}, which "
                             f"{module} does not define; pass model_type.")
    else:
        raise ValueError("The checkpoint records no model class; pass model_type.")
    if class_path:
        import_class_module(class_path)
    source = os.fspath(path)
    try:
        model = cls.load_from_checkpoint(source, map_location=map_location, **overrides)
    except Exception as exc:
        failed = class_path_in(str(exc), cls.__name__) or class_path
        if not failed or failed.rpartition(".")[0] in sys.modules:
            raise
        if not import_class_module(failed):
            raise UnimportableClassError(failed) from exc
        model = cls.load_from_checkpoint(source, map_location=map_location, **overrides)
    hparams = getattr(model, "hparams", None)
    if not isinstance(hparams, dict) or not hparams:
        hparams = record["hyper_parameters"]
    info = CheckpointInfo(format="lightning", arch=cls.__name__,
                          hyper_parameters=dict(hparams), class_path=class_path)
    return _finish(model, map_location), info


def load_checkpoint(path, model_type: str | None = None, map_location="cpu", *,
                    payload: Any = _UNREAD, overrides: dict | None = None):
    """(model, CheckpointInfo) for an inference or a Lightning checkpoint.

    The format is told by content: what torch.load(weights_only=True)
    reads is an inference checkpoint when it is a dict with arch,
    hyper_parameters and state_dict, and a Lightning checkpoint otherwise.
    model_type, when given, must be a class in tfbs.nn.models and, for an
    inference checkpoint, equal its arch (else ValueError); without it a
    Lightning checkpoint is loaded as the class its hyper_parameters
    _class_path names.  overrides replace hyper_parameters (Lightning
    passes them to load_from_checkpoint).  payload is what read_payload
    returned, when the caller has read the file already; for a Lightning
    checkpoint {"hyper_parameters": ...} is enough.

    The model is returned in eval mode on map_location (None leaves it
    where loading put it).  Raises ValueError for a model_type or arch
    that does not fit and for a file that is a bare state_dict or no
    PyTorch checkpoint (see checkpoint_problem), UnimportableClassError
    for a class module that cannot be imported, and whatever building or
    loading the model raises.
    """
    overrides = dict(overrides or {})
    if payload is _UNREAD:
        payload = read_payload(path)
    problem = checkpoint_problem(path, payload)
    if problem:
        raise ValueError(problem)
    if is_inference_payload(payload):
        return _load_inference(payload, model_type, map_location, overrides)
    # Lightning's save_hyperparameters keeps the calling frames alive in a
    # reference cycle until the next garbage collection, so no frame on the
    # stack may still hold the payload: torch.load(mmap=True) keeps the file
    # mapped, which on Windows blocks overwriting the checkpoint.
    record = _lightning_record(payload)
    del payload
    return _load_lightning(path, record, model_type, map_location, overrides)
