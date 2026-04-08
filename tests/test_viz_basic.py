import sys
import types
import importlib
import os

# Create minimal mock for tangermeme submodules used by viz.py so tests are lightweight
mock_tangermeme = types.SimpleNamespace()
mock_utils = types.SimpleNamespace(
    random_one_hot=lambda shape: __import__("torch").rand(shape),
    one_hot_encode=lambda seq: __import__("torch").nn.functional.one_hot(
        __import__("torch").tensor([0]*len(seq)), num_classes=4).permute(1,0).float()[:,:len(seq)],
    pwm_consensus=lambda pwm: __import__("torch").tensor([[0.25]*pwm.shape[1]]*4)
)
mock_predict = types.SimpleNamespace(predict=lambda model, X, device=None: __import__("torch").rand(X.shape[0],1))
mock_plot = types.SimpleNamespace(plot_logo=lambda *a, **k: None)
mock_ersatz = types.SimpleNamespace(substitute=lambda *a, **k: None)
mock_marginalize = types.SimpleNamespace(marginalize=lambda *a, **k: (None, None))
mock_io = types.SimpleNamespace(read_meme=lambda *a, **k: {}, extract_loci=lambda *a, **k: __import__("torch").rand(1,4,24))
mock_seqlet = types.SimpleNamespace(recursive_seqlets=lambda *a, **k: [])
mock_annotate = types.SimpleNamespace(annotate_seqlets=lambda *a, **k: ([],[]), count_annotations=lambda *a, **k: __import__("torch").zeros(1))
mock_deep = types.SimpleNamespace(deep_lift_shap=lambda *a, **k: __import__("torch").zeros(1,4,24), _captum_deep_lift_shap=lambda *a, **k: __import__("torch").zeros(1,4,24), hypothetical_attributions=lambda *a, **k: None)
mock_variant = types.SimpleNamespace(substitution_effect=lambda *a, **k: None, deletion_effect=lambda *a, **k: None, insertion_effect=lambda *a, **k: None)
mock_ism = types.SimpleNamespace(saturation_mutagenesis=lambda *a, **k: __import__("torch").zeros(1,4,24))

sys.modules['tangermeme'] = types.ModuleType('tangermeme')
sys.modules['tangermeme.utils'] = mock_utils
sys.modules['tangermeme.predict'] = mock_predict
sys.modules['tangermeme.plot'] = mock_plot
sys.modules['tangermeme.ersatz'] = mock_ersatz
sys.modules['tangermeme.marginalize'] = mock_marginalize
sys.modules['tangermeme.io'] = mock_io
sys.modules['tangermeme.seqlet'] = mock_seqlet
sys.modules['tangermeme.annotate'] = mock_annotate
sys.modules['tangermeme.deep_lift_shap'] = mock_deep
sys.modules['tangermeme.variant_effect'] = mock_variant
sys.modules['tangermeme.ism'] = mock_ism

# Also ensure captum is minimally present if viz imports it
sys.modules['captum'] = types.ModuleType('captum')
sys.modules['captum.attr'] = types.ModuleType('captum.attr')

# Ensure the scripts directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
viz = importlib.import_module("scripts.viz")

def test_get_consensus_sequence_exact_match():
    dp = viz.DataPreparer(24, device=__import__("torch").device("cpu"))
    seq = dp.get_consensus_sequence()
    assert len(seq) == 24
    assert seq == "CACTGCCCAGTCAAGTGTTCTTGA"

def test_get_consensus_sequence_truncate():
    dp = viz.DataPreparer(50, device=__import__("torch").device("cpu"))
    seq = dp.get_consensus_sequence()
    assert len(seq) == 50

def test_cli_parsing_analyses():
    argv = ["viz.py", "--model-path", "m", "--model-type", "BPNetReal", "--analyses", "sequence,mutagenesis", "--test"]
    old = sys.argv
    sys.argv = argv
    cfg = viz.AnalysisConfig()
    sys.argv = old
    assert "sequence" in cfg.analyses
    assert "mutagenesis" in cfg.analyses