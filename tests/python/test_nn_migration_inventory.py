"""The migration ledger must account for all canonical examples without wrappers."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_output_comparison_checks_nested_variant_histories_and_missing_arrays(tmp_path):
    spec = importlib.util.spec_from_file_location("compare_nn_outputs", ROOT / "scripts/compare_nn_outputs.py")
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    for language in comparison.LANGUAGES:
        directory = tmp_path / language / "quick/models/autoencoder/competitive"
        directory.mkdir(parents=True)
        (directory / "best.npz").write_bytes(b"identical persisted arrays")
        (directory / "history.tsv").write_text("epoch\tloss\n1\t0.5\n")
    assert comparison.compare(tmp_path, ["autoencoder"], include_history=True)["passed"]
    r_directory = tmp_path / "r/quick/models/autoencoder/competitive"
    (r_directory / "history.tsv").write_text("epoch\tloss\n1\t0.6\n")
    report = comparison.compare(tmp_path, ["autoencoder"], include_history=True)
    assert not report["passed"]
    assert not report["models"]["autoencoder"]["files"]["competitive/history.tsv"]["byte_identical"]
    assert comparison.compare(tmp_path, ["autoencoder"])["passed"]
    (r_directory / "best.npz").unlink()
    assert not comparison.compare(tmp_path, ["autoencoder"])["passed"]
    assert not comparison.compare(tmp_path, ["missing_model"])["passed"]


def test_complete_canonical_inventory_and_honest_implementation_status():
    report = json.loads((ROOT / "examples/nn/language_status.json").read_text())
    canonical = {p.stem: p for p in (ROOT / "examples/cpp/source/nn").glob("[12]_*/**/*.cpp")}
    canonical.update(
        {p.stem: p for p in (ROOT / "examples/nn").glob("[0-9]*/**/*.cpp")}
    )
    assert set(report["examples"]) == set(canonical)
    assert report["total"] == len(canonical)
    for name, record in report["examples"].items():
        assert ROOT / record["cpp"] == canonical[name]
        for language in ("python", "r"):
            path = ROOT / record[language]["path"]
            state = record[language]["implementation"]
            if state == "native":
                assert path.read_text().startswith("# Native ")
                assert "run_nn_example(" not in path.read_text()
                if record["classification"]:
                    text = path.read_text()
                    assert "confusion" in text.lower(), (name, language)
            elif state == "reference_wrapper":
                assert path.exists() and not path.read_text().startswith("# Native ")
            else:
                assert state == "missing" and not path.exists()


def test_language_command_names_use_native_launchers():
    python = subprocess.run(
        ["just", "--dry-run", "python-example", "cnn_mnist", "epochs=5"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "scripts/python_example.py" in python.stderr + python.stdout
    assert "DATAMUNGE_BUILD_NN_EXAMPLES=0" in python.stderr + python.stdout
    r = subprocess.run(
        ["just", "--dry-run", "r-example", "cnn_mnist", "epochs=5"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Rscript --vanilla -e" in r.stderr + r.stdout
    assert "python" not in (r.stderr + r.stdout).lower()


@pytest.mark.parametrize("dependency", [
    "examples/bindings/nn_components.i",
    "src/pydatamunge/nn_components.py",
    "examples/nn/common/model.hpp",
    "examples/python/nn/common/model.py",
    "examples/r/nn/common/model.R",
])
def test_shared_component_changes_invalidate_validation_evidence(tmp_path, monkeypatch, dependency):
    spec = importlib.util.spec_from_file_location("migration_status", ROOT / "scripts/nn_migration_status.py")
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    monkeypatch.setattr(migration, "ROOT", tmp_path)
    monkeypatch.setattr(migration, "STATUS", tmp_path / "examples/nn/language_status.json")
    files = {
        "prebindings/nn/src/nn.i": "",
        "examples/bindings/nn_components.i": "",
        "src/pydatamunge/nn.py": "",
        "src/pydatamunge/nn_components.py": "",
        "src/pydatamunge/nn_workflows.py": "",
        "prebindings/nn/r/R/nn.R": "",
        "examples/r/nn/common.R": "",
        "examples/nn/common/model.hpp": "",
        "examples/python/nn/common/model.py": "",
        "examples/r/nn/common/model.R": "",
        "examples/cpp/source/nn/1_demo/model.cpp": "int main() {}",
        "examples/python/nn/1_demo/model.py": "# Native Python NN example\n",
        "examples/r/nn/1_demo/model.R": "# Native R NN example\n",
    }
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    report = migration.refresh(write_catalog=False)
    entry = report["examples"]["model"]
    for language in ("python", "r"):
        entry[language]["validation"] = {"status": "passed"}
    entry["cpp_reference_validation"] = {"status": "passed"}
    migration.STATUS.write_text(json.dumps(report))
    unchanged = migration.refresh(write_catalog=False)["examples"]["model"]
    assert unchanged["python"]["validation"] and unchanged["cpp_reference_validation"]
    (tmp_path / dependency).write_text("changed")
    changed = migration.refresh(write_catalog=False)["examples"]["model"]
    assert changed["python"]["validation"] is None and changed["r"]["validation"] is None
    assert "cpp_reference_validation" not in changed
