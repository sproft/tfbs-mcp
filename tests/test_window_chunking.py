"""Window-mode scoring stays below the 32-bit index limit of CUDA kernels.

On 19 October 2026, tfbs_predict in window mode crashed on an A100 with "CUDA error: an illegal
memory access was encountered": a batch of 512 sequences of 500 bp gives 512 x 477 = 244,224
windows of 24 bp in one forward pass, and the NKX2.1 VCNNBpnet models pool each window to
num_channels x pool_output_size values, 3.3e9 to 4.4e9 elements in all, above 2**31.
These tests check that every forward pass is now split below MAX_ACTIVATION_ELEMENTS and that
the split does not change the scores.
"""
import asyncio
import json

import pytest
import torch

from tfbs.nn.models import VCNNBpnet
from tfbs.prediction.analysis import (
    MAX_ACTIVATION_ELEMENTS,
    find_best_windows_fast,
    forward_in_chunks,
    largest_activation_per_sample,
    safe_forward_batch_size,
)

INT32_LIMIT = 2 ** 31


def _vcnn(num_channels=8, kernel_size=13, pool_output_size=178, dilations=(1, 1, 2), dense=(16, 8)):
    torch.manual_seed(0)
    model = VCNNBpnet(input_length=24, num_channels=num_channels, kernel_size=kernel_size,
                      dilations=list(dilations), pool_output_size=pool_output_size, dense_sizes=list(dense))
    return model.eval()


def _onehot(n, length, seed=1):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 4, (n, length), generator=g)
    return torch.nn.functional.one_hot(idx, 4).float().permute(0, 2, 1).contiguous()


class _Functional(torch.nn.Module):
    """Builds its largest activation, 4 x 24 x 50 per input, in functional code."""

    def __init__(self, hidden=3):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(4 * 24 * 50, hidden)

    def forward(self, x):
        return self.lin(x.repeat(1, 1, 50).flatten(1))


class _Squeezed(torch.nn.Module):
    """Returns shape (N,), which a batch of one squeezes to a scalar."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.lin(x.mean(dim=2)).squeeze()


def test_largest_activation_is_the_pooled_output():
    model = _vcnn(num_channels=8, pool_output_size=178)
    # the pooled and flattened output, 8 x 178, is larger than the input (4 x 24),
    # the convolutions (8 x 24) and the dense layers
    assert largest_activation_per_sample(model, _onehot(2, 24)) == 8 * 178


def test_functional_intermediates_count():
    # forward hooks see only the module outputs (3 per input); the repeated input is 4800
    assert largest_activation_per_sample(_Functional(), _onehot(1, 24)) == 4 * 24 * 50


def test_weights_that_do_not_grow_with_the_batch_do_not_count():
    # the 4800 x 1000 weight (and its transposed view) is the largest tensor of the pass,
    # but it does not grow with the batch
    model = _Functional(hidden=1000).eval()
    assert largest_activation_per_sample(model, _onehot(1, 24)) == 4 * 24 * 50


@pytest.mark.filterwarnings("ignore:.*torch.jit.script.*:DeprecationWarning")
def test_torchscript_models_are_measured():
    model = _Functional().eval()
    scripted = torch.jit.script(model)
    assert largest_activation_per_sample(scripted, _onehot(1, 24)) == 4 * 24 * 50


def test_forward_hooks_are_the_fallback_without_dispatch_modes(monkeypatch):
    from tfbs.prediction import analysis

    monkeypatch.setattr(analysis, "_TorchDispatchMode", None)
    assert largest_activation_per_sample(_vcnn(num_channels=8, pool_output_size=178), _onehot(1, 24)) == 8 * 178


def test_probe_restores_modes_and_leaves_batchnorm_statistics_alone():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv1d(4, 8, 3), torch.nn.BatchNorm1d(8), torch.nn.Flatten())
    model.train()
    model[1].eval()  # a mixed state, restored module by module
    before = model[1].running_mean.clone(), model[1].num_batches_tracked.clone()
    largest_activation_per_sample(model, _onehot(1, 24))
    assert model.training and model[0].training and not model[1].training
    assert torch.equal(model[1].running_mean, before[0])
    assert torch.equal(model[1].num_batches_tracked, before[1])


def test_safe_batch_size_respects_the_cap():
    model = _vcnn(num_channels=8, pool_output_size=178)
    per = 8 * 178
    assert safe_forward_batch_size(model, _onehot(2, 24), max_elements=10_000) == 10_000 // per
    assert safe_forward_batch_size(model, _onehot(2, 24), max_elements=1e4) == 10_000 // per  # float cap
    assert safe_forward_batch_size(model, _onehot(2, 24), max_elements=10) == 1  # never 0


def test_forward_in_chunks_matches_one_pass():
    model = _vcnn()
    x = _onehot(50, 24)
    with torch.no_grad():
        full = model(x)
        split = forward_in_chunks(model, x, 7)
    assert split.shape == full.shape
    assert torch.allclose(split, full, atol=1e-6)


def test_forward_in_chunks_handles_a_last_slice_squeezed_to_a_scalar():
    model = _Squeezed().eval()
    x = _onehot(7, 24)
    with torch.no_grad():
        full = model(x)
        split = forward_in_chunks(model, x, 3)  # slices of 3, 3 and 1
    assert full.shape == split.shape == (7,)
    assert torch.allclose(split, full, atol=1e-6)


def test_window_mode_scores_do_not_depend_on_the_split():
    model = _vcnn()
    X = _onehot(9, 60)
    ref = find_best_windows_fast(X, model, window_size=24, processing_batch_size=512,
                                 device="cpu", max_activation_elements=10 ** 12)
    split = find_best_windows_fast(X, model, window_size=24, processing_batch_size=512,
                                   device="cpu", max_activation_elements=10_000)
    best_w_ref, best_s_ref, all_ref, mean_ref = ref
    best_w, best_s, all_s, mean_s = split
    assert all_s.shape == all_ref.shape == (9, 60 - 24 + 1)
    assert torch.allclose(torch.from_numpy(all_s), torch.from_numpy(all_ref), atol=1e-6)
    assert torch.allclose(torch.from_numpy(best_s), torch.from_numpy(best_s_ref), atol=1e-6)
    assert torch.allclose(torch.from_numpy(mean_s), torch.from_numpy(mean_ref), atol=1e-6)
    assert (best_w == best_w_ref).all()


def test_window_mode_forward_passes_stay_under_the_cap():
    model = _vcnn(num_channels=8, pool_output_size=178)
    per = 8 * 178
    cap = 10_000
    sizes = []
    handle = model.register_forward_pre_hook(lambda _m, inp: sizes.append(inp[0].shape[0]))
    try:
        find_best_windows_fast(_onehot(20, 60), model, window_size=24, processing_batch_size=512,
                               device="cpu", max_activation_elements=cap)
    finally:
        handle.remove()
    # two probe passes of 2 and 4 windows, then 20 x 37 = 740 windows in passes of at most cap // per
    assert sizes[:2] == [2, 4]
    assert sum(sizes[2:]) == 20 * 37
    assert max(sizes[2:]) * per <= cap


def test_prediction_script_scores_do_not_depend_on_the_split(monkeypatch):
    for name in ("logomaker", "tangermeme", "pyfaidx"):
        pytest.importorskip(name)
    from scripts import prediction
    from tfbs.prediction import analysis

    model = _vcnn()
    X = _onehot(6, 50)
    ref = prediction.extract_best_windows(X, model, window_size=24, batch_size=512, device="cpu")
    calls = []
    real = analysis.safe_forward_batch_size

    def five_windows(m, sample, *a, **k):
        calls.append(sample.shape[0])
        real(m, sample, *a, **k)
        return 5

    monkeypatch.setattr(analysis, "safe_forward_batch_size", five_windows)
    sizes = []
    handle = model.register_forward_pre_hook(lambda _m, inp: sizes.append(inp[0].shape[0]))
    try:
        split = prediction.extract_best_windows(X, model, window_size=24, batch_size=512, device="cpu")
    finally:
        handle.remove()
    assert calls == [1]
    assert max(sizes[2:]) == 5 and sum(sizes[2:]) == 6 * 27
    for a, b in zip(ref, split):
        assert torch.allclose(torch.as_tensor(a).float(), torch.as_tensor(b).float(), atol=1e-6)


@pytest.mark.parametrize("channels,kernel,pool", [(82, 13, 178), (120, 13, 151), (82, 42, 167)])
def test_nkx21_models_would_overflow_without_the_split(channels, kernel, pool):
    model = _vcnn(num_channels=channels, kernel_size=kernel, pool_output_size=pool,
                  dilations=(1, 1, 2, 4, 8), dense=(128, 64))
    per = largest_activation_per_sample(model, _onehot(2, 24))
    assert per == channels * pool
    windows_at_default_batch = 512 * (500 - 24 + 1)
    assert windows_at_default_batch * per > INT32_LIMIT  # what crashed on 19 October 2026
    safe = safe_forward_batch_size(model, _onehot(2, 24))
    assert safe * per <= MAX_ACTIVATION_ELEMENTS < INT32_LIMIT


def test_server_prediction_uses_the_split(monkeypatch):
    from tfbs.mcp import server

    server._ensure_imports()
    model = _vcnn()
    server._model_registry["chunk_test"] = {"model": model, "model_type": "VCNNBpnet", "device": "cpu",
                                            "input_length": 24, "classify": False}
    seqs = ["ACGTTGCAACGTAGCTAGCTAGGA" * 3, "TTGACCATGGCATCGATCGGATCC" * 3]
    try:
        plain = json.loads(asyncio.run(server.tfbs_predict(model_id="chunk_test", sequences=[s[:24] for s in seqs])))
        window = json.loads(asyncio.run(server.tfbs_predict(model_id="chunk_test", sequences=seqs,
                                                            window_mode=True, window_size=24)))
        calls = []
        real = server._safe_forward_batch_size

        def one_row(m, sample, *a, **k):
            calls.append(sample.shape[0])
            real(m, sample, *a, **k)
            return 1

        monkeypatch.setattr(server, "_safe_forward_batch_size", one_row)
        plain_split = json.loads(asyncio.run(server.tfbs_predict(model_id="chunk_test", sequences=[s[:24] for s in seqs])))
    finally:
        server._model_registry.pop("chunk_test", None)
    assert calls == [1]
    assert plain_split["scores"] == pytest.approx(plain["scores"], abs=1e-6)
    assert len(window["best_scores"]) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_window_mode_at_full_size_on_gpu():
    model = _vcnn(num_channels=120, kernel_size=13, pool_output_size=151, dilations=(1, 1, 2, 4, 8), dense=(128, 64))
    X = _onehot(512, 500)
    best_w, best_s, all_s, mean_s = find_best_windows_fast(X, model, window_size=24, processing_batch_size=512,
                                                           device="cuda")
    assert all_s.shape == (512, 477)
    assert best_s.shape == mean_s.shape == (512,)
