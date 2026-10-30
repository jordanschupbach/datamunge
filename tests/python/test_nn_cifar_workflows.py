"""Run actual bound CNN updates through the native scripts with a small data fixture."""
import importlib.util
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from pydatamunge import nn

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIRS = {
    "senet": "se_resnet50_compact_cifar10", "eca_net": "eca_resnet50_compact_cifar10",
    "alexnet": "alexnet_compact_cifar10", "xception": "xception_compact_cifar10",
    "densenet": "densenet_bc40_k12_cifar10", "ghostnet": "ghostnet_cifar10",
    "mobilenet_v2": "mobilenet_v2_cifar10", "mnasnet": "mnasnet_a1_cifar10",
    "convnext": "convnext_atto_cifar10", "efficientnet": "efficientnet_b0_cifar10",
    "efficientnet_v2": "efficientnet_v2_b0_cifar10",
    "preactivation_resnet": "preactivation_resnet164_compact_cifar10",
    "pyramidnet": "pyramidnet110_a48_cifar10", "res2net": "res2net50_compact_cifar10",
    "resnext": "resnext29_8x4d_compact_cifar10", "revnet": "revnet38_cifar10",
    "cbam": "cbam_resnet50_compact_cifar10", "non_local": "nonlocal_resnet50_compact_cifar10",
    "octconv": "oct_resnet50_compact_cifar10", "repvgg": "repvgg_a0_cifar10",
    "sknet": "sknet50_compact_cifar10",
}


@pytest.mark.parametrize("name", MODEL_DIRS)
def test_native_cifar_loop_honors_epochs_and_restores_selected_checkpoint(name, tmp_path, monkeypatch, capsys):
    script = next((ROOT / "examples/python/nn").rglob(name + ".py"))
    code = runpy.run_path(str(script))
    main = code["main"]
    # Keep the 5k-row split contract while fitting just two rows. Models,
    # autograd, augmentation, Adam, evaluation and disk checkpoints stay real.
    development, test = nn.ImageDataset(), nn.ImageDataset()
    development.images = nn.zeros([5002, 32, 32, 3])
    development.labels = nn.zeros([5002], nn.DType.Int64)
    test.images = nn.ones([3, 32, 32, 3])
    test.labels = nn.from_vector_int64([1, 2, 3], [3])
    monkeypatch.setenv("DATAMUNGE_EXAMPLES_FULL", "0")
    monkeypatch.setenv("DATAMUNGE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setitem(main.__globals__, "full_checks", lambda: None)
    if name + "_checks" in main.__globals__:
        checks = main.__globals__[name + "_checks"]
        # Full-size numerical diagnostics are exercised by port validation.
        # Keep constructors, model forwards and all actual fitting operations.
        members = {key: (lambda *args: None) if key.endswith(("Check", "Checks")) else value
                   for key, value in vars(checks).items()}
        monkeypatch.setitem(main.__globals__, name + "_checks", SimpleNamespace(**members))
    monkeypatch.setattr(code["image_training"], "LoadCifar10", lambda _path: (development, test))
    previous = nn.get_num_threads()
    try:
        assert main(["epochs=3", "fit=2", "batch=2", "threads=2"]) == 0
    finally:
        nn.set_num_threads(previous)
    directory = tmp_path / "models" / MODEL_DIRS[name]
    history = (directory / "history.tsv").read_text().splitlines()
    assert [line.split("\t")[0] for line in history] == ["epoch", "1", "2", "3"]
    assert (directory / "best_params.npz").is_file()
    assert (directory / "best_state.npz").is_file() == (name not in ("alexnet", "convnext"))
    output = capsys.readouterr().out
    assert "selected epoch=" in output and "confusion (truth rows)" in output
    # The example checks that the restored loss equals its selected validation
    # loss, and that confusion contains exactly all three test rows.


def test_python_launcher_scopes_mode_flags_and_restores_environment(monkeypatch):
    spec = importlib.util.spec_from_file_location("python_example", ROOT / "scripts/python_example.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    from pydatamunge import nn_examples
    seen = []
    monkeypatch.setenv("DATAMUNGE_EXAMPLES_FULL", "1")
    monkeypatch.setenv("DATAMUNGE_OUTPUT_DIR", "preserved")
    monkeypatch.setattr(nn_examples, "NATIVE_EXAMPLES", ("resnet_bottleneck",))
    monkeypatch.setattr(runpy, "run_path", lambda path: {
        "main": lambda args: seen.append((args, os.environ["DATAMUNGE_EXAMPLES_FULL"])) or 0
    })
    monkeypatch.setattr("sys.argv", ["python_example.py", "resnet_bottleneck", "--quick", "epochs=3"])
    assert runner.main() == 0
    assert seen == [(["epochs=3"], "0")]
    assert os.environ["DATAMUNGE_EXAMPLES_FULL"] == "1"
    assert os.environ["DATAMUNGE_OUTPUT_DIR"] == "preserved"
