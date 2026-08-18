"""Tests for scripts/viz.py.

The script depends on tangermeme, captum, and seaborn submodules that
are heavy to install in CI. We mock them via a module-scoped fixture
that installs the mocks, imports viz, then restores ``sys.modules`` so
later test files (e.g. tests/test_mcp_tools.py) still see the real
tangermeme implementation when they import it.
"""
import importlib
import os
import sys
import types

import pytest


_MOCK_KEYS = [
    "tangermeme",
    "tangermeme.utils",
    "tangermeme.predict",
    "tangermeme.plot",
    "tangermeme.ersatz",
    "tangermeme.marginalize",
    "tangermeme.io",
    "tangermeme.seqlet",
    "tangermeme.annotate",
    "tangermeme.deep_lift_shap",
    "tangermeme.variant_effect",
    "tangermeme.ism",
    "tangermeme.saturation_mutagenesis",
    "captum",
    "captum.attr",
    "seaborn",
    "scripts.viz",
]


def _build_mocks():
    import torch

    mock_utils = types.SimpleNamespace(
        random_one_hot=lambda shape: torch.rand(shape),
        one_hot_encode=lambda seq: torch.nn.functional.one_hot(
            torch.tensor([0] * len(seq)), num_classes=4
        ).permute(1, 0).float()[:, : len(seq)],
        pwm_consensus=lambda pwm: torch.tensor([[0.25] * pwm.shape[1]] * 4),
    )
    mock_predict = types.SimpleNamespace(
        predict=lambda model, X, device=None: torch.rand(X.shape[0], 1)
    )
    mock_plot = types.SimpleNamespace(plot_logo=lambda *a, **k: None)
    mock_ersatz = types.SimpleNamespace(substitute=lambda *a, **k: None)
    mock_marginalize = types.SimpleNamespace(marginalize=lambda *a, **k: (None, None))
    mock_io = types.SimpleNamespace(
        read_meme=lambda *a, **k: {},
        extract_loci=lambda *a, **k: torch.rand(1, 4, 24),
    )
    mock_seqlet = types.SimpleNamespace(recursive_seqlets=lambda *a, **k: [])
    mock_annotate = types.SimpleNamespace(
        annotate_seqlets=lambda *a, **k: ([], []),
        count_annotations=lambda *a, **k: torch.zeros(1),
    )
    mock_deep = types.SimpleNamespace(
        deep_lift_shap=lambda *a, **k: torch.zeros(1, 4, 24),
        _captum_deep_lift_shap=lambda *a, **k: torch.zeros(1, 4, 24),
        hypothetical_attributions=lambda *a, **k: None,
    )
    mock_variant = types.SimpleNamespace(
        substitution_effect=lambda *a, **k: None,
        deletion_effect=lambda *a, **k: None,
        insertion_effect=lambda *a, **k: None,
    )
    mock_ism = types.SimpleNamespace(
        saturation_mutagenesis=lambda *a, **k: torch.zeros(1, 4, 24)
    )

    mock_captum = types.ModuleType("captum")
    mock_captum_attr = types.ModuleType("captum.attr")
    mock_captum_attr.DeepLiftShap = type("DeepLiftShap", (), {})
    mock_captum_attr.InputXGradient = type("InputXGradient", (), {})

    mock_seaborn = types.SimpleNamespace(set_style=lambda *a: None)

    return {
        "tangermeme": types.ModuleType("tangermeme"),
        "tangermeme.utils": mock_utils,
        "tangermeme.predict": mock_predict,
        "tangermeme.plot": mock_plot,
        "tangermeme.ersatz": mock_ersatz,
        "tangermeme.marginalize": mock_marginalize,
        "tangermeme.io": mock_io,
        "tangermeme.seqlet": mock_seqlet,
        "tangermeme.annotate": mock_annotate,
        "tangermeme.deep_lift_shap": mock_deep,
        "tangermeme.variant_effect": mock_variant,
        "tangermeme.ism": mock_ism,
        "tangermeme.saturation_mutagenesis": mock_ism,
        "captum": mock_captum,
        "captum.attr": mock_captum_attr,
        "seaborn": mock_seaborn,
    }


@pytest.fixture(scope="module")
def viz():
    """Install mock tangermeme/captum/seaborn, import scripts.viz, tear down."""
    saved = {k: sys.modules.get(k) for k in _MOCK_KEYS}
    mocks = _build_mocks()
    for k, v in mocks.items():
        sys.modules[k] = v

    scripts_parent = os.path.join(os.path.dirname(__file__), "..")
    if scripts_parent not in sys.path:
        sys.path.insert(0, scripts_parent)
    sys.modules.pop("scripts.viz", None)
    viz_module = importlib.import_module("scripts.viz")

    yield viz_module

    for k in _MOCK_KEYS:
        if saved[k] is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = saved[k]


def test_get_consensus_sequence_exact_match(viz):
    import torch
    dp = viz.DataPreparer(24, device=torch.device("cpu"))
    seq = dp.get_consensus_sequence()
    assert len(seq) == 24
    assert seq == "CACTGCCCAGTCAAGTGTTCTTGA"


def test_get_consensus_sequence_truncate(viz):
    import torch
    dp = viz.DataPreparer(50, device=torch.device("cpu"))
    seq = dp.get_consensus_sequence()
    assert len(seq) == 50


def test_cli_parsing_analyses(viz):
    argv = [
        "viz.py",
        "--model-path", "m",
        "--model-type", "BPNetReal",
        "--analyses", "sequence,mutagenesis",
        "--test",
    ]
    old = sys.argv
    sys.argv = argv
    try:
        cfg = viz.AnalysisConfig()
    finally:
        sys.argv = old
    assert "sequence" in cfg.analyses
    assert "mutagenesis" in cfg.analyses
