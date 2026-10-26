"""The migration ledger must account for all canonical examples without wrappers."""

import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def test_complete_canonical_inventory_and_honest_implementation_status():
    report = json.loads((ROOT / "examples/nn/language_status.json").read_text())
    canonical = {p.stem: p for p in (ROOT / "examples/cpp/nn").glob("[12]_*/**/*.cpp")}
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
                    assert "Confusion matrix" in text or "print_confusion(" in text
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
