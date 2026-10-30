"""Real memory-model training, controls and checkpoints through native workflows."""
import json
from pathlib import Path
import runpy

import pytest
from pydatamunge import nn, nn_components

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["ntm", "dnc", "memnet", "key_value"])
def test_memory_example_runs_complete_workflow(tmp_path, monkeypatch, name):
    data = tmp_path / "datasets"
    nn_components.memory_components.PrepareData(str(data))
    monkeypatch.setenv("DATAMUNGE_DATA_DIR", str(data))
    output = tmp_path / name
    source = ROOT / "examples/python/nn/4_recurrent_and_sequence_models/4.3_memory_augmented_networks" / (name + ".py")
    program = runpy.run_path(str(source))
    assert program["main"](["epochs=3", "out=" + str(output), "tf="]) == 0
    metric = json.loads((output / "metrics.json").read_text())
    assert metric["epochs"] == 3 and metric["updates"] == 48
    assert len((output / "history.tsv").read_text().splitlines()) == 4
    first = dict(nn.load_npz(str(output / "initial_arrays.npz")))
    final = dict(nn.load_npz(str(output / "model_arrays.npz")))
    assert first.keys() == final.keys()
    assert any(not nn.equal_bits(first[key], final[key]) for key in first)
    assert (output / "padding_outputs.npz").exists()
    if name in ("memnet", "key_value"):
        assert (output / "empty_outputs.npz").exists() and (output / "shuffled_outputs.npz").exists()
    else:
        fixture = "address_fixture.npz" if name == "ntm" else "access_fixture.npz"
        assert (output / fixture).exists()
    assert (output / "index.html").exists()
