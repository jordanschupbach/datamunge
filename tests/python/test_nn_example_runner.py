"""Regression checks for checkpoint dependencies in the example runner."""

import contextlib
import importlib.util
import io
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "nn_example_runner",
    Path(__file__).resolve().parents[2] / "scripts/run_nn_examples.py",
)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_checkpoint_producer_precedes_consumers_once():
    assert runner.execution_order(["deepdream", "vanilla_saliency", "grad_cam"]) == [
        "vanilla_saliency",
        "deepdream",
        "grad_cam",
    ]
    assert runner.execution_order(["receptive_fields", "cnn_mnist"]) == [
        "cnn_mnist",
        "receptive_fields",
    ]


@pytest.mark.parametrize(
    "producer,consumer",
    [("vanilla_saliency", "deepdream"), ("cnn_mnist", "receptive_fields")],
)
def test_missing_checkpoint_invalidates_producer_and_consumer(
    tmp_path, producer, consumer
):
    names = [producer, consumer]
    checkpoint = tmp_path / "model_arrays.npz"
    cache = {name: ("source", "library") for name in names}
    with (
        mock.patch.object(sys, "argv", ["run_nn_examples.py", "--list", consumer]),
        mock.patch.object(
            runner, "examples", return_value={name: Path(name) for name in names}
        ),
        mock.patch.object(runner, "load_cache", return_value=cache),
        mock.patch.object(runner, "library_key", return_value="library"),
        mock.patch.object(runner, "source_key", return_value="source"),
        mock.patch.object(runner, "REQUIRED_OUTPUTS", {producer: (checkpoint,)}),
    ):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assert runner.main() == 0
        assert all("changed" in line for line in output.getvalue().splitlines())
        checkpoint.touch()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assert runner.main() == 0
        assert all("up to date" in line for line in output.getvalue().splitlines())


def test_deadline_stops_process_group_and_does_not_pass(tmp_path):
    binary = tmp_path / "example"
    binary.touch()
    process = mock.Mock(pid=12345)
    process.wait.side_effect = [subprocess.TimeoutExpired(str(binary), 0.1), 0]
    with (
        mock.patch.object(runner.subprocess, "Popen", return_value=process) as launch,
        mock.patch.object(runner.os, "killpg") as stop,
    ):
        assert not runner.run_example(binary, timeout=0.1)
        assert launch.call_args.kwargs["start_new_session"]
        stop.assert_called_once_with(12345, signal.SIGTERM)


def test_real_sleeping_process_is_bounded():
    started = time.monotonic()
    assert not runner.run_example(
        Path(sys.executable), ("-c", "import time; time.sleep(30)"), timeout=0.2
    )
    assert time.monotonic() - started < 3


@pytest.mark.parametrize("profile", ["quick", "full"])
def test_training_profiles_have_separate_caches(profile):
    arguments = ["run_nn_examples.py", "--list"] + (
        ["--full"] if profile == "full" else []
    )
    with (
        mock.patch.object(sys, "argv", arguments),
        mock.patch.object(
            runner, "examples", return_value={"example": Path("example")}
        ),
        mock.patch.object(runner, "load_cache", return_value={}) as load,
        mock.patch.object(runner, "source_key", return_value="source"),
        mock.patch.object(runner, "library_key", return_value="library"),
    ):
        assert runner.main() == 0
        load.assert_called_once_with(runner.CACHE.with_name(f"run-cache-{profile}.tsv"))


def test_failed_producer_prevents_parallel_consumer_run(tmp_path):
    names = {name: Path(name) for name in ("cnn_mnist", "receptive_fields")}
    with (
        mock.patch.object(
            sys,
            "argv",
            ["run_nn_examples.py", "--force", "--jobs", "2", "receptive_fields"],
        ),
        mock.patch.object(runner, "examples", return_value=names),
        mock.patch.object(runner, "CACHE", tmp_path / "run-cache.tsv"),
        mock.patch.object(runner, "load_cache", return_value={}),
        mock.patch.object(runner, "save_cache"),
        mock.patch.object(runner, "source_key", return_value="source"),
        mock.patch.object(runner, "library_key", return_value="library"),
        mock.patch.object(runner, "run_example", return_value=False) as run,
    ):
        assert runner.main() == 1
        assert run.call_count == 1
        assert run.call_args.args[0].name == "cnn_mnist"


def test_single_example_build_includes_only_its_prerequisites():
    with (
        mock.patch.object(
            sys, "argv", ["run_nn_examples.py", "--build-only", "receptive_fields"]
        ),
        mock.patch.object(
            runner.subprocess, "run", return_value=mock.Mock(returncode=0)
        ) as build,
        mock.patch.object(runner, "load_cache") as load,
    ):
        assert runner.main() == 0
        command = build.call_args.args[0]
        assert command[command.index("--target") + 1 :] == [
            "cnn_mnist",
            "receptive_fields",
        ]
        load.assert_not_called()


@pytest.mark.parametrize("override", [False, True])
def test_output_base_and_override_keep_checkpoint_paths_consistent(
    tmp_path, monkeypatch, override
):
    base = tmp_path / "python-output"
    output = tmp_path / "custom" if override else base / "quick"
    checkpoint = output / "models/cnn_mnist/best.npz"
    monkeypatch.setenv("DATAMUNGE_EXAMPLES_FULL", "0")
    monkeypatch.setenv("DATAMUNGE_OUTPUT_DIR", str(output) if override else "")
    cache = {}
    with (
        mock.patch.object(runner, "OUTPUT_BASE", base),
        mock.patch.object(runner, "OUTPUT_ROOT", base / "quick"),
        mock.patch.object(
            runner,
            "REQUIRED_OUTPUTS",
            {"cnn_mnist": (base / "quick/models/cnn_mnist/best.npz",)},
        ),
        mock.patch.object(runner, "CACHE", tmp_path / "cache.tsv"),
        mock.patch.object(
            runner, "examples", return_value={"cnn_mnist": Path("cnn_mnist.cpp")}
        ),
        mock.patch.object(runner, "source_key", return_value="source"),
        mock.patch.object(runner, "library_key", return_value="library"),
        mock.patch.object(runner, "load_cache", return_value=cache),
        mock.patch.object(runner, "save_cache"),
        mock.patch.object(runner, "run_example", return_value=True) as execute,
    ):
        with mock.patch.object(
            sys, "argv", ["run_nn_examples.py", "--force", "cnn_mnist"]
        ):
            assert runner.main() == 0
        assert execute.call_args.args[3]["DATAMUNGE_OUTPUT_DIR"] == str(output)
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        execute.reset_mock()
        with mock.patch.object(sys, "argv", ["run_nn_examples.py", "cnn_mnist"]):
            assert runner.main() == 0
        execute.assert_not_called()
