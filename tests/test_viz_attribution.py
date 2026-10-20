"""scripts/viz.py attributions (Captum path): pairs, eval mode, reproducible baselines.

run_deep_lift_shap used to pass every input repeated s times, with the s baselines tiled,
to captum's DeepLiftShap, which pairs every input row with every baseline row: one call
ran the model on 2 x (batch_N x s)**2 sequences, up to 131,072, above the 32-bit index
limit of CUDA kernels for the NKX2.1 VCNNBpnet models.  It also put the whole model in
train mode, so Dropout was active and BatchNorm updated its statistics, and it drew the
baselines from fresh entropy, so random_state did not make a call reproducible.
"""
import importlib
import os
import sys

import numpy as np
import pytest
import torch

for _name in ("captum", "tangermeme", "seaborn"):
    pytest.importorskip(_name)


@pytest.fixture(scope="module")
def viz():
    root = os.path.join(os.path.dirname(__file__), "..")
    if root not in sys.path:
        sys.path.insert(0, root)
    saved = sys.modules.pop("scripts.viz", None)
    module = importlib.import_module("scripts.viz")
    yield module
    sys.modules.pop("scripts.viz", None)
    if saved is not None:
        sys.modules["scripts.viz"] = saved


class _Net(torch.nn.Module):
    def __init__(self, dropout=0.0, batchnorm=False, length=12):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv1d(4, 6, 5, padding=2)
        self.bn = torch.nn.BatchNorm1d(6) if batchnorm else torch.nn.Identity()
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dropout)
        self.flat = torch.nn.Flatten()
        self.fc = torch.nn.Linear(6 * length, 1)

    def forward(self, x):
        return self.fc(self.flat(self.drop(self.relu(self.bn(self.conv(x))))))


class _Recurrent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.gru = torch.nn.GRU(4, 3, num_layers=2, dropout=0.3, batch_first=True)
        self.drop = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(3, 1)

    def forward(self, x):
        out, _ = self.gru(x.permute(0, 2, 1))
        return self.fc(self.drop(out[:, -1]))


def _onehot(n, length=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 4, (n, length), generator=g)
    return torch.nn.functional.one_hot(idx, 4).float().permute(0, 2, 1).contiguous()


def _core(viz, model):
    return viz.AttributionCore(model, torch.device("cpu"), tangermeme=False)


def _baselines(viz, seed, s, length=12):
    return viz.random_one_hot((s, 4, length), random_state=np.random.RandomState(seed)).float()


def test_pairs_give_captums_all_pairs_result(viz):
    from captum.attr import DeepLiftShap
    from tangermeme.deep_lift_shap import hypothetical_attributions

    model = _Net().eval()
    X = _onehot(5)
    core = _core(viz, model)
    attr = core.run_deep_lift_shap(X, n_shuffles=6, random_state=3, verbose=False)
    hyp = core.run_deep_lift_shap(X, n_shuffles=6, random_state=3, verbose=False, hypothetical=True)
    # the old call on the same 6 baselines (5 inputs and 6 shuffles fit in one call):
    # every repeated input row averaged over all 30 baseline rows
    block = _baselines(viz, 3, 6)
    old = DeepLiftShap(model).attribute(X.repeat_interleave(6, dim=0), baselines=block.repeat(5, 1, 1),
                                        target=0, custom_attribution_func=hypothetical_attributions)
    old_hyp = old.detach().reshape(5, 6, 4, 12).sum(dim=1) / 6
    assert hyp.shape == attr.shape == X.shape
    assert torch.allclose(hyp, old_hyp, atol=1e-5)
    assert torch.allclose(attr, old_hyp * X, atol=1e-5)


def test_raw_outputs_are_the_multipliers_of_each_pair(viz):
    from captum.attr import DeepLift

    model = _Net().eval()
    X = _onehot(5)
    raw = _core(viz, model).run_deep_lift_shap(X, n_shuffles=6, random_state=3, verbose=False, raw_outputs=True)
    block = _baselines(viz, 3, 6)
    expected = DeepLift(model).attribute(X.repeat_interleave(6, dim=0), baselines=block.repeat(5, 1, 1),
                                         target=0, custom_attribution_func=lambda m, i, b: m)
    assert raw.shape == (5, 6, 4, 12)
    assert torch.allclose(raw, expected.detach().reshape(5, 6, 4, 12), atol=1e-6)
    # one value per shuffle, not their mean copied into every slot
    assert not torch.allclose(raw[:, 0], raw[:, 1])


def test_no_forward_pass_holds_more_than_two_rows_per_pair(viz):
    model = _Net().eval()
    sizes = []
    handle = model.conv.register_forward_pre_hook(lambda _m, inp: sizes.append(inp[0].shape[0]))
    try:
        _core(viz, model).run_deep_lift_shap(_onehot(300), n_shuffles=4, random_state=0, verbose=False)
    finally:
        handle.remove()
    # 256 inputs x 1 shuffle per call, then 44 inputs x 4 shuffles; each pair runs as an input
    # row and a baseline row, and every one of the 300 x 4 pairs runs exactly once
    assert max(sizes) <= 2 * 256
    assert sum(sizes) == 2 * 300 * 4


def test_dropout_and_batchnorm_stay_in_eval_mode(viz):
    model = _Net(dropout=0.5, batchnorm=True)
    model.train()
    model.bn.eval()  # a mixed state, restored module by module
    stats = model.bn.running_mean.clone(), model.bn.num_batches_tracked.clone()
    seen = []
    handle = model.drop.register_forward_pre_hook(lambda m, _inp: seen.append(m.training))
    core = _core(viz, model)
    X = _onehot(7)
    try:
        first = core.run_deep_lift_shap(X, n_shuffles=5, random_state=1, verbose=False)
        again = core.run_deep_lift_shap(X, n_shuffles=5, random_state=1, verbose=False)
    finally:
        handle.remove()
    assert seen and not any(seen)
    assert torch.equal(first, again)
    assert torch.equal(model.bn.running_mean, stats[0]) and torch.equal(model.bn.num_batches_tracked, stats[1])
    assert model.training and model.drop.training and not model.bn.training


def test_random_state_makes_a_call_reproducible(viz):
    core = _core(viz, _Net().eval())
    X = _onehot(4)
    a = core.run_deep_lift_shap(X, n_shuffles=8, random_state=11, verbose=False)
    b = core.run_deep_lift_shap(X, n_shuffles=8, random_state=11, verbose=False)
    c = core.run_deep_lift_shap(X, n_shuffles=8, random_state=12, verbose=False)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_input_x_gradient_keeps_only_recurrent_layers_in_train_mode(viz):
    model = _Recurrent()
    model.train()
    seen = {}

    def record(name, value):
        seen.setdefault(name, value)  # returns None, so the hook leaves the input alone

    hooks = [model.gru.register_forward_pre_hook(lambda m, _inp: record("gru", (m.training, m.dropout))),
             model.drop.register_forward_pre_hook(lambda m, _inp: record("drop", m.training))]
    core = _core(viz, model)
    X = _onehot(6)
    try:
        first = core.calculate_input_x_gradient(X)
        again = core.calculate_input_x_gradient(X)
    finally:
        for h in hooks:
            h.remove()
    assert seen["gru"] == (True, 0.0)  # train mode for cuDNN, without inter-layer dropout
    assert seen["drop"] is False
    assert torch.equal(first, again)
    assert model.training and model.gru.training and model.gru.dropout == 0.3
