"""The VCNNBpnet template: a three-layer default, the study's five layers on request.

tfbs/configs/models/VCNNBpnet.yaml carries the NKX2-1 study's tuned
settings (do_train_final.sh, all_mean, 24 bp) except the depth: dilations
[1, 1, 2] instead of the study's [1, 1, 2, 4, 8].  As a classifier on small
sets at batch size 64 the five-layer stack can stay at a constant output
(loss ln 2, AUC 0.5); its output std at initialisation is about 1e-6,
that of the three-layer stack about 1e-4.  tfbs_config's
model_params={"dilations": [1, 1, 2, 4, 8]} restores the study's stack, and
checkpoints of that stack keep loading, because the loader rebuilds a model
from the hyper_parameters stored in the checkpoint, not from the template.
"""
import asyncio
import json
import random
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import yaml

from tfbs.mcp import server
from tfbs.nn import models
from tfbs.nn.loading import load_checkpoint

TEMPLATE = Path(server._model_templates()["VCNNBpnet"])
STUDY_DILATIONS = [1, 1, 2, 4, 8]
# The other values of the template are the study's (do_train_final.sh, all_mean).
STUDY_VALUES = {"input_length": 24, "input_channels": 4, "num_channels": 82,
                "kernel_size": 42, "pool_output_size": 167, "dense_sizes": [128, 64]}
MOTIF = "CACTTGA"


def run(coro):
    return json.loads(asyncio.run(coro))


@pytest.fixture(autouse=True)
def _cleanup():
    server._ensure_imports()
    yield
    for key in [k for k in server._model_registry if k.startswith("vt_")]:
        server._model_registry.pop(key)


def _template_args() -> dict:
    with open(TEMPLATE, encoding="utf-8") as fh:
        return dict(yaml.safe_load(fh)["init_args"])


def _classifier(seed: int, **overrides) -> models.VCNNBpnet:
    torch.manual_seed(seed)
    return models.VCNNBpnet(classify=True, learning_rate=1e-3,
                            **{**_template_args(), **overrides})


def _one_hot(seqs: list[str]) -> torch.Tensor:
    idx = torch.tensor([["ACGT".index(b) for b in s] for s in seqs])
    return F.one_hot(idx, num_classes=4).float().permute(0, 2, 1).contiguous()


def _random_one_hot(n: int, length: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 4, (n, length), generator=gen)
    return F.one_hot(idx, num_classes=4).float().permute(0, 2, 1).contiguous()


def _planted_motif_set(n: int, length: int = 24, seed: int = 0):
    """Half positives with the motif at a random position, half negatives with a shuffle of it.

    Both classes have the same base composition, so only the motif tells them apart.
    """
    rng = random.Random(seed)
    rc = MOTIF[::-1].translate(str.maketrans("ACGT", "TGCA"))
    seqs, labels = [], []
    for i in range(n):
        insert = MOTIF
        if i % 2:
            while insert in (MOTIF, rc):
                insert = "".join(rng.sample(MOTIF, len(MOTIF)))
        seq = [rng.choice("ACGT") for _ in range(length)]
        start = rng.randrange(length - len(insert) + 1)
        seq[start:start + len(insert)] = insert
        seqs.append("".join(seq))
        labels.append(float(i % 2 == 0))
    order = list(range(n))
    rng.shuffle(order)
    x = _one_hot([seqs[i] for i in order])
    y = torch.tensor([labels[i] for i in order]).unsqueeze(1)
    return x, y


# ---------------------------------------------------------------------------
# The template
# ---------------------------------------------------------------------------

def test_template_has_the_three_layer_stack_and_the_study_values():
    with open(TEMPLATE, encoding="utf-8") as fh:
        template = yaml.safe_load(fh)
    assert template["class_path"] == "tfbs.nn.models.VCNNBpnet"
    args = template["init_args"]
    assert args["dilations"] == [1, 1, 2]
    assert {k: v for k, v in args.items() if k != "dilations"} == STUDY_VALUES
    text = TEMPLATE.read_text(encoding="utf-8")
    assert 'model_params={"dilations": [1, 1, 2, 4, 8]}' in text
    assert "do_train_final.sh" in text


def test_only_the_template_changed_and_the_class_default_keeps_five_layers():
    import inspect
    default = inspect.signature(models.VCNNBpnet.__init__).parameters["dilations"].default
    assert default == STUDY_DILATIONS
    assert "class default is still [1, 1, 2, 4, 8]" in " ".join(
        line.lstrip("# ") for line in TEMPLATE.read_text(encoding="utf-8").splitlines())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_default_classifier_output_is_not_constant_at_initialisation(seed):
    x = _random_one_hot(512, 24, seed=7)
    three = _classifier(seed).eval()
    five = _classifier(seed, dilations=STUDY_DILATIONS).eval()
    assert len(three.conv_layers) == 3 and len(five.conv_layers) == 5
    with torch.no_grad():
        three_std = three(x).std().item()
        five_std = five(x).std().item()
    assert three_std > 5e-5, three_std
    # The study's five-layer stack starts about two orders of magnitude flatter.
    assert five_std < 5e-5, five_std


def test_default_classifier_learns_an_easy_planted_motif_in_two_epochs():
    # 4,000 training and 1,000 validation sequences; about 5 s on a desktop CPU.
    x, y = _planted_motif_set(5000, seed=1234)
    x_val, y_val, x_train, y_train = x[:1000], y[:1000], x[1000:], y[1000:]
    model = _classifier(0)
    torch.manual_seed(0)
    optimizer = model.configure_optimizers()
    for _ in range(2):
        model.train()
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), 64):
            batch = perm[i:i + 64]
            loss, _, _ = model._handle_batch((x_train[batch], y_train[batch]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        p = model.eval()(x_val)
    auc = torchmetrics.functional.auroc(p.reshape(-1), y_val.reshape(-1).long(),
                                        task="binary").item()
    assert auc > 0.95, auc


# ---------------------------------------------------------------------------
# The study's five-layer stack through tfbs_config and the loaders
# ---------------------------------------------------------------------------

def _dataset(tmp_path: Path, n: int = 16) -> Path:
    root = tmp_path / "bound_dataset"
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)
        torch.save(_random_one_hot(n, 24, seed=1), root / split / "seqs.pt")
        torch.save(torch.tensor([[1.0], [0.0]] * (n // 2)), root / split / "labels.pt")
    return root


def _study_config(tmp_path: Path) -> tuple[dict, dict]:
    data = _dataset(tmp_path)
    out = tmp_path / "study.yaml"
    result = run(server.tfbs_config(model_type="VCNNBpnet", data_path=str(data),
                                    output_path=str(out),
                                    model_params={"dilations": STUDY_DILATIONS}))
    return result, yaml.safe_load(out.read_text())["model"]["init_args"]


def test_tfbs_config_writes_the_three_layer_default(tmp_path):
    data = _dataset(tmp_path)
    out = tmp_path / "default.yaml"
    result = run(server.tfbs_config(model_type="VCNNBpnet", data_path=str(data),
                                    output_path=str(out)))
    assert "error" not in result, result
    args = yaml.safe_load(out.read_text())["model"]["init_args"]
    assert args["dilations"] == [1, 1, 2] == result["model_init_args"]["dilations"]


def test_tfbs_config_model_params_restore_the_study_stack(tmp_path):
    result, args = _study_config(tmp_path)
    assert "error" not in result, result
    assert args["dilations"] == STUDY_DILATIONS
    assert {k: args[k] for k in STUDY_VALUES} == STUDY_VALUES
    assert args["classify"] is True
    assert len(models.VCNNBpnet(**args).conv_layers) == 5


def test_tool_description_names_the_default_and_the_override_in_one_sentence():
    desc = " ".join(server.tfbs_config.__doc__.split())
    sentence = next(s for s in desc.split(". ") if "three-layer" in s)
    assert "[1, 1, 2]" in sentence
    assert 'model_params={"dilations": [1, 1, 2, 4, 8]}' in sentence


def test_a_five_layer_lightning_checkpoint_still_loads(tmp_path):
    _, args = _study_config(tmp_path)
    torch.manual_seed(0)
    model = models.VCNNBpnet(**args)
    x, y = _planted_motif_set(64, seed=5)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=32)
    trainer = pl.Trainer(max_steps=2, accelerator="cpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, limit_val_batches=0)
    trainer.fit(model, train_dataloaders=loader)
    ckpt = tmp_path / "ckpt" / "best_VCNNBpnet.ckpt"
    trainer.save_checkpoint(ckpt)
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved["hyper_parameters"]["dilations"] == STUDY_DILATIONS

    probe = _random_one_hot(8, 24, seed=3)
    with torch.no_grad():
        expected = model.eval()(probe)

    loaded, info = load_checkpoint(ckpt, "VCNNBpnet")
    assert info.format == "lightning"
    assert info.hyper_parameters["dilations"] == STUDY_DILATIONS
    assert len(loaded.conv_layers) == 5
    with torch.no_grad():
        np.testing.assert_allclose(loaded(probe).numpy(), expected.numpy(), rtol=1e-5, atol=1e-7)

    result = run(server.tfbs_load_model("vt_five", str(ckpt), "VCNNBpnet", device="cpu"))
    assert "error" not in result, result
    assert result["classify"] is True and result["input_length"] == 24
    served = run(server.tfbs_predict("vt_five", sequences=["ACGT" * 6]))
    with torch.no_grad():
        reference = model(server._sequences_to_tensor(["ACGT" * 6])).flatten()[0].item()
    assert served["scores"][0] == pytest.approx(reference, rel=1e-5, abs=1e-7)
