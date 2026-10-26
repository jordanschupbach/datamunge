"""The migrated C++ catalogue and Python/SWIG replicas stay in lockstep."""

import importlib.util
import os
import runpy
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from pydatamunge import nn_examples

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "nn_example_generator", ROOT / "scripts/prebuild_nn_examples.py"
)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_complete_migrated_catalogue_and_python_launchers(monkeypatch):
    sources = generator.examples()
    assert len(sources) == 165
    assert {p.stem for p in sources} <= set(nn_examples.available(native_only=False))
    assert not list((ROOT / "examples/nn").glob("[12]_*/**/*.cpp"))
    calls = []
    monkeypatch.setattr(
        nn_examples, "run", lambda name, args: calls.append((name, args)) or 0
    )
    for source in sources:
        script = (
            ROOT
            / "examples/python/nn"
            / source.relative_to(generator.CPP_ROOT).with_suffix(".py")
        )
        if script.read_text().startswith("# Native "):
            continue
        entrypoint = runpy.run_path(str(script))
        assert entrypoint["main"](["epochs=1"]) == 0
        assert calls[-1] == (source.stem, ["epochs=1"])


def test_environment_and_original_arguments_are_restored(monkeypatch, tmp_path):
    import pydatamunge

    observed = []

    def entry(name, args):
        observed.append(
            (
                name,
                args,
                os.environ["DATAMUNGE_EXAMPLES_FULL"],
                os.environ["DATAMUNGE_OUTPUT_DIR"],
            )
        )
        raise RuntimeError("native fixture failure")

    monkeypatch.setattr(
        pydatamunge,
        "_nn_examples",
        SimpleNamespace(run_nn_example=entry),
        raising=False,
    )
    monkeypatch.setenv("DATAMUNGE_EXAMPLES_FULL", "0")
    monkeypatch.setenv("DATAMUNGE_OUTPUT_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="native fixture failure"):
        nn_examples.run_reference("alexnet", ["--full", "epochs=1"])
    assert observed == [("alexnet", ["epochs=1"], "1", str(tmp_path))]
    assert os.environ["DATAMUNGE_EXAMPLES_FULL"] == "0"
    assert os.environ["DATAMUNGE_OUTPUT_DIR"] == str(tmp_path)
    with pytest.raises(ValueError, match="unknown NN example"):
        nn_examples.run("missing", [])


def test_embedded_cpp_entrypoints_keep_main_semantics_and_private_symbols(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    first = tmp_path / "first.cpp"
    first.write_text("""#include <string>
int helper() { return 4; }
int main() { std::string braces = R"raw({ ignore braces })raw"; /* } */
  if (helper() != 4) return 1;
}
""")
    second = tmp_path / "second.cpp"
    second.write_text("""#include <string>
int helper() { return 7; }
int main(int argc, char** argv) { return argc + helper(); }
""")
    generated = []
    for source in (first, second):
        target = tmp_path / ("generated_" + source.name)
        target.write_text(generator.translation_unit(source))
        generated.append(str(target))
    driver = tmp_path / "driver.cpp"
    driver.write_text("""int datamunge_example_entry_first(int, char**);
int datamunge_example_entry_second(int, char**);
int main() {
  return datamunge_example_entry_first(1, nullptr) == 0 &&
         datamunge_example_entry_second(3, nullptr) == 10 ? 0 : 1;
}
""")
    executable = tmp_path / "entrypoints"
    subprocess.run(
        [compiler, "-std=c++17", *generated, str(driver), "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_native_catalogue_and_data_free_numerical_fixture(monkeypatch):
    native = pytest.importorskip("pydatamunge._nn_examples")
    assert set(native.nn_example_names()) == {p.stem for p in generator.examples()}

    def reject_subprocess(*args, **kwargs):
        raise AssertionError("an NN binding call must not launch a subprocess")

    monkeypatch.setattr(subprocess, "run", reject_subprocess)
    monkeypatch.setattr(subprocess, "Popen", reject_subprocess)
    assert nn_examples.run("convolution_im2col", []) == 0
    with pytest.raises(RuntimeError, match="unknown NN example"):
        native.run_nn_example("missing", [])
    with pytest.raises(RuntimeError, match="NUL"):
        native.run_nn_example("cnn_mnist", ["bad\0argument"])


def test_python_runner_uses_installed_binding_despite_pythonpath(tmp_path):
    import sys

    pytest.importorskip("pydatamunge._nn_examples")
    shadow = tmp_path / "shadow/pydatamunge"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text(
        'raise RuntimeError("stale source package selected")\n'
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(shadow.parent),
        DATAMUNGE_OUTPUT_DIR=str(tmp_path / "outputs"),
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_nn_python_examples.py"),
            "--force",
            "convolution_im2col",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All im2col convolution checks passed" in result.stdout
    for output, expected in [
        (tmp_path / "outputs", "up to date"),
        (tmp_path / "different-output", "changed"),
    ]:
        listed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_nn_python_examples.py"),
                "--list",
                "convolution_im2col",
            ],
            cwd=ROOT,
            env=dict(env, DATAMUNGE_OUTPUT_DIR=str(output)),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert listed.returncode == 0, listed.stdout + listed.stderr
        assert expected in listed.stdout


def test_r_catalogue_and_launchers_match_python():
    import re

    catalogue = (ROOT / "examples/bindings/r/R/catalog.R").read_text()
    assert set(re.findall(r'"([^"\n]+)"', catalogue)) == {p.stem for p in generator.examples()}
    for source in generator.examples():
        script = ROOT / "examples/r/nn" / source.relative_to(generator.CPP_ROOT).with_suffix(".R")
        text = script.read_text()
        if text.startswith('# Native '):
            continue
        assert f'nn_example_run("{source.stem}", commandArgs(TRUE))' in text
        assert 'quit(status = status)' in text
