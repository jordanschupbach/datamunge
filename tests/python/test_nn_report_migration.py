"""Report retirement needs complete execution evidence and a retained replacement."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def migration(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("nn_report_migration", ROOT / "scripts/nn_report_migration.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "LEDGER", tmp_path / "ledger.json")
    source = tmp_path / "examples/nn/0_foundations/0.1_demo/demo.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("int main() {}\n")
    report = tmp_path / "docs/mepml/examples/nn/0_foundations/0.1_demo/demo.mepml"
    report.parent.mkdir(parents=True)
    report.write_text('```{cpp}\nint main() {}\n```\n')
    evidence = tmp_path / "run.json"
    evidence.write_text(json.dumps({"files": [{"file": str(report), "failed": 0,
                         "blocks": [{"lang": "cpp", "status": "ok"}]}]}))
    return module, source, report, evidence


def test_retirement_preserves_digest_and_report(migration):
    module, source, report, evidence = migration
    records = module.inventory()
    original = module.digest(source)
    module.retire(records, "demo", evidence)
    assert not source.exists() and report.is_file()
    entry = records["0_foundations/0.1_demo/demo"]
    assert entry["source_sha256"] == original and entry["retired"]
    assert entry["verification"]["report_sha256"] == module.digest(report)
    module.LEDGER.write_text(json.dumps({"examples": records}))
    assert module.inventory() == records
    report.unlink()
    with pytest.raises(ValueError, match="no replacement"):
        module.inventory()


@pytest.mark.parametrize("kind", ["failed", "skipped", "missing", "different_report"])
def test_incomplete_evidence_never_removes_original(migration, kind):
    module, source, report, evidence = migration
    data = json.loads(evidence.read_text())
    entry = data["files"][0]
    if kind in ("failed", "skipped"):
        entry["blocks"][0]["status"] = kind
    elif kind == "missing":
        entry["blocks"] = []
    else:
        entry["file"] = str(report.with_name("other.mepml"))
    evidence.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.retire(module.inventory(), "demo", evidence)
    assert source.is_file()


def test_truncated_runner_results_cannot_retire_a_two_block_report(migration):
    module, source, report, evidence = migration
    report.write_text(report.read_text() * 2)
    with pytest.raises(ValueError, match="every runnable block"):
        module.retire(module.inventory(), "demo", evidence)
    assert source.is_file()


def test_checked_in_retirement_ledger_accounts_for_every_original():
    spec = importlib.util.spec_from_file_location("nn_report_migration", ROOT / "scripts/nn_report_migration.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorded = json.loads(module.LEDGER.read_text())
    assert module.inventory() == recorded["examples"]
    assert len(recorded["examples"]) == recorded["total"]
    assert sum(r["retired"] for r in recorded["examples"].values()) == recorded["retired_sources"]
