"""非训练检查：用合成数据与替身训练步骤验证接线，禁止优化器更新。

uv run python -m stage4_transformer_encoder.test_train_wiring
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import DataLoader

from stage3_variable_length.data import TinyCaptchaDataset
from stage3_variable_length.models import DEFAULT_CHARACTERS

from . import analyze_validation, train
from .models import CaptchaTransformer


def fake_dataset(path):
    data = TinyCaptchaDataset(["aa", "a7"], seed=0)
    data.split = "train" if path == train.TRAIN_DIR else "validation"
    data.characters = DEFAULT_CHARACTERS
    data.width, data.height = 180, 100
    data.records = [
        {"label": label, "font": __file__, "file": f"{label}.png"}
        for label in data.labels
    ]
    return data


def check_rerender():
    data = fake_dataset(train.TRAIN_DIR)
    original = data[0][0].clone()
    disabled = train.AugmentedDataset(data, augment=False)
    rng_before = disabled.rng.getstate()
    assert torch.equal(disabled[0][0], original)
    assert disabled.rng.getstate() == rng_before
    for probability in (-1, 2, float("nan")):
        try:
            train.AugmentedDataset(data, rerender_probability=probability)
        except ValueError:
            pass
        else:
            raise AssertionError("非法重渲染概率应被拒绝")

    def render(label, font, rng, *, width, height):
        assert label == "aa" and font == Path(__file__)
        return Image.new("RGB", (width, height), tuple(rng.randrange(256) for _ in range(3)))

    with patch.object(train, "render_variable_length_captcha", side_effect=render) as renderer:
        first = train.AugmentedDataset(data, augment=False, rerender_probability=1, seed=7)
        second = train.AugmentedDataset(data, augment=False, rerender_probability=1, seed=7)
        image, label = first[0]
        assert label == "aa" and image.shape == (3, 100, 180)
        assert image.dtype == torch.float32 and 0 <= image.min() <= image.max() <= 1
        assert torch.equal(image, second[0][0])
        assert not torch.equal(image, first[0][0])
        mixed = train.AugmentedDataset(data, rerender_probability=0.5)
        with patch.object(mixed.rng, "random", side_effect=[0.9, 0.1]), patch.object(
            train, "affine", side_effect=lambda image, *a, **kw: image
        ) as transform:
            assert torch.equal(mixed[0][0], original)
            assert not torch.equal(mixed[0][0], original)
            assert transform.call_count == 2
        assert renderer.call_count == 4
    assert torch.equal(data[0][0], original) and len(first) == len(data)
    for records in ([{"label": "aa"}], [{"label": "aa", "font": __file__ + ".missing"}]):
        data.records = records
        try:
            train.AugmentedDataset(data, rerender_probability=1)
        except ValueError:
            pass
        else:
            raise AssertionError("缺少字体应被拒绝")


def demo():
    check_rerender()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    model = CaptchaTransformer().eval()
    wider = CaptchaTransformer(sequence_width=45).eval()
    assert model.sequence_width == 22 and wider.sequence_width == 45
    assert sum(p.numel() for p in wider.parameters()) == 141141
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 100, 180)).shape == (2, 22, 21)
        logits = wider(torch.zeros(2, 3, 100, 180))
        assert logits.shape == (2, 45, 21)
        loss = torch.nn.CTCLoss(blank=20)(
            logits.log_softmax(-1).transpose(0, 1),
            torch.tensor([0, 0, 1, 1]), torch.tensor([45, 45]), torch.tensor([2, 2]),
        )
        assert torch.isfinite(loss)
    smaller = CaptchaTransformer(ffn_dim=64).eval()
    assert sum(p.numel() for p in smaller.parameters()) < sum(
        p.numel() for p in model.parameters()
    )
    assert model.use_position_encoding is True
    no_position = CaptchaTransformer(use_position_encoding=False).eval()
    no_position.load_state_dict(model.state_dict(), strict=True)
    images = torch.randn(2, 3, 100, 180)
    with torch.no_grad():
        features = no_position.bottleneck(no_position.features(images))
        raw_sequence = features.mean(dim=2).transpose(1, 2)
        torch.testing.assert_close(
            no_position(images),
            no_position.classifier(no_position.encoder(raw_sequence)),
        )
        assert not torch.equal(model(images), no_position(images))
    assert sum(p.numel() for p in no_position.parameters()) == 141141
    regularized = CaptchaTransformer(dropout=0.1).eval()
    regularized.load_state_dict(model.state_dict(), strict=True)
    sequence = torch.randn(2, 22, 64)
    with torch.no_grad():
        torch.testing.assert_close(
            model.encoder(sequence), regularized.encoder(sequence)
        )
        regularized.encoder.train()
        assert not torch.equal(
            regularized.encoder(sequence), regularized.encoder(sequence)
        )
    assert sum(p.numel() for p in regularized.parameters()) == 141141
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loader = DataLoader(fake_dataset(train.TRAIN_DIR), batch_size=2)
    calibrated = train.recalibrate_batchnorm(model, loader, torch.device("cpu"))
    calibrated_dropout = train.recalibrate_batchnorm(
        regularized, loader, torch.device("cpu")
    )
    assert not calibrated_dropout.encoder.dropout.training
    assert calibrated is not model and not calibrated.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    for name, parameter in calibrated.named_parameters():
        torch.testing.assert_close(parameter, before[name])
    assert any(
        not torch.equal(value, before[key])
        for key, value in calibrated.state_dict().items()
        if "running_mean" in key
    )
    try:
        train.recalibrate_batchnorm(model, [], torch.device("cpu"))
    except ValueError:
        pass
    else:
        raise AssertionError("空校准集应被拒绝")
    for arguments in (
        ["--epochs", "0"],
        ["--lr", "nan"],
        ["--batch-size", "0"],
        ["--dropout", "nan"],
        ["--dropout", "1"],
        ["--dropout", "-0.1"],
        ["--rerender-probability", "nan"],
        ["--rerender-probability", "1.1"],
        ["--sequence-width", "30"],
    ):
        with redirect_stdout(StringIO()), patch("sys.stderr", new=StringIO()):
            try:
                train.parse_args(arguments)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError("非法参数应被拒绝")
    assert train.parse_args([]).device == "auto"
    assert train.parse_args([]).dropout == 0.0
    assert train.parse_args([]).ffn_dim == 128
    assert train.parse_args([]).length_6_loss_weight == 2.0
    assert train.parse_args(["--new-renders-only"]).new_renders_only is True
    assert train.parse_args([]).use_position_encoding is True
    assert train.parse_args([]).norm_first is False
    assert train.parse_args([]).rerender_probability == 0
    assert train.parse_args([]).sequence_width == 22
    args = train.parse_args(
        [
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--dropout",
            "0.1",
            "--no-position-encoding",
            "--pre-norm",
            "--ffn-dim",
            "64",
            "--length-6-loss-weight",
            "3",
            "--augment",
            "--rerender-probability",
            "0.5",
            "--sequence-width",
            "45",
        ]
    )
    diagnostics = SimpleNamespace(
        average_loss=0.0, nonfinite_loss_count=0, nonfinite_gradient_count=0
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.bin"
        with (
            patch.object(train, "parse_args", return_value=args),
            patch.object(train, "ManifestCaptchaDataset", side_effect=fake_dataset),
            patch.object(train, "MODEL_PATH", path),
            patch.object(train, "recalibrate_batchnorm", wraps=train.recalibrate_batchnorm) as bn,
            patch.object(train, "evaluate_model", wraps=train.evaluate_model) as evaluate,
            patch.object(
                train, "run_training_epoch_with_diagnostics", return_value=diagnostics
            ) as step,
            patch.object(
                torch.optim.Adam, "step", side_effect=AssertionError("禁止训练")
            ),
            redirect_stdout(StringIO()),
        ):
            train.main()
        step.assert_called_once()
        assert step.call_args.kwargs["length_6_loss_weight"] == 3.0
        assert isinstance(step.call_args.args[1].dataset, train.AugmentedDataset)
        assert not isinstance(bn.call_args.args[1].dataset, train.AugmentedDataset)
        assert all(not isinstance(call.args[1].dataset, train.AugmentedDataset)
                   for call in evaluate.call_args_list)
        checkpoint = torch.load(path, weights_only=True)
        assert checkpoint["epoch"] == 1
        assert checkpoint["model_config"]["sequence_model"] == "transformer"
        assert checkpoint["training_config"]["recalibrate_batchnorm"] is True
        assert checkpoint["training_config"]["length_6_loss_weight"] == 3.0
        assert checkpoint["training_config"]["rerender_probability"] == 0.5
        assert checkpoint["model_config"]["dropout"] == 0.1
        assert checkpoint["model_config"]["use_position_encoding"] is False
        assert checkpoint["model_config"]["norm_first"] is True
        assert checkpoint["model_config"]["ffn_dim"] == 64
        assert checkpoint["model_config"]["sequence_width"] == 45
        restored = CaptchaTransformer(
            checkpoint["characters"],
            dropout=checkpoint["model_config"]["dropout"],
            use_position_encoding=checkpoint["model_config"]["use_position_encoding"],
            norm_first=checkpoint["model_config"]["norm_first"],
            ffn_dim=checkpoint["model_config"]["ffn_dim"],
            sequence_width=checkpoint["model_config"]["sequence_width"],
        ).eval()
        restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
        images, _ = next(iter(loader))
        with torch.inference_mode():
            assert restored(images).shape == (2, 45, 21)
        with (
            patch("sys.argv", ["analyze_validation", "--device", "cpu", "--decoder", "beam", "--beam-width", "2"]),
            patch.object(analyze_validation, "MODEL_PATH", path),
            patch.object(analyze_validation, "ManifestCaptchaDataset", side_effect=fake_dataset),
            patch.object(analyze_validation, "write_validation_reports", return_value=(path, path)) as report,
            redirect_stdout(StringIO()),
        ):
            analyze_validation.main()
        assert report.call_args.args[0]["total_sequences"] == 2
        assert report.call_args.args[0]["decoder"] == "beam"
        assert report.call_args.args[0]["beam_width"] == 2
        assert report.call_args.args[2].name == "beam_2"
        assert not path.with_suffix(".tmp").exists()
    print("非训练检查通过：重渲染分支、随机复现、BN/验证原图隔离、参数校验、checkpoint 加载。")


if __name__ == "__main__":
    demo()
