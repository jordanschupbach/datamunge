"""Docs setup failures stop before reports run; nested preparation keeps root paths."""
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("website_report_check", ROOT / "scripts/website/reports.py")
reports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reports)


def document(path, code="#include <dm/dm.hpp>\nint main() {}"):
    """Create a report using the same package flags as the real C++ reports."""
    path.write_text('```{cpp, flags="-std=c++23 $(pkg-config --cflags --libs datamunge) '
                    '-Wl,-rpath,$(pkg-config --variable=libdir datamunge)"}\n' + code + '\n```\n')
    return str(path)


def test_report_collections_keep_stable_urls_and_discover_both(tmp_path):
    current, earlier = map(Path, reports.example_roots(tmp_path))
    for collection, name in [(current, "graph/search/new"), (earlier, "math/old")]:
        source = collection / (name + ".mepml")
        source.parent.mkdir(parents=True)
        source.write_text("Report")
    found = reports.discover_reports(tmp_path)
    assert [reports.report_rel(p, [current, earlier]) for p in found] == [
        "graph/search/new.mepml", "math/old.mepml"]
    assert reports.Report(found[0], reports.report_rel(found[0], [current, earlier])).url == (
        "reports/graph/search/new.html")


def test_report_collections_reject_duplicate_site_paths(tmp_path):
    for collection in reports.example_roots(tmp_path):
        source = Path(collection) / "same.mepml"
        source.parent.mkdir(parents=True)
        source.write_text("Report")
    with pytest.raises(reports.ToolchainError, match="Duplicate report path same.mepml"):
        reports.discover_reports(tmp_path)


@pytest.mark.parametrize("collection", ["docs/examples/mepml", "docs/mepml/examples"])
def test_report_source_link_uses_its_actual_collection(tmp_path, monkeypatch, collection):
    pytest.importorskip("markdown_it", reason="website rendering uses the docs shell dependencies")
    monkeypatch.setitem(sys.modules, "reports", reports)
    spec = importlib.util.spec_from_file_location("website_source_link_check", ROOT / "scripts/website/build.py")
    website = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(website)
    monkeypatch.setattr(website, "ROOT", str(tmp_path))
    source = tmp_path / collection / "graph/search/example.mepml"
    report = reports.Report(str(source), "graph/search/example.mepml")
    site = website.Site.__new__(website.Site)
    site.report_uses = {}
    site.source_url = lambda path: "https://example.test/source/" + path
    site.reports_sidebar = lambda *args: ""
    rendered = []
    site.write = lambda page, title, content, *args, **kwargs: rendered.append((page, content))
    site.write_report(report)
    assert rendered[0][0] == "reports/graph/search/example.html"
    assert f'https://example.test/source/{collection}/graph/search/example.mepml' in rendered[0][1]


@pytest.fixture
def package(tmp_path, monkeypatch):
    """Build a small real shared library and package to isolate discovery/link checks."""
    compiler = shutil.which("c++")
    if not compiler or not shutil.which("pkg-config"):
        pytest.skip("C++ compiler and pkg-config are required")
    include = tmp_path / "include/dm"
    include.mkdir(parents=True)
    header = include / "dm.hpp"
    header.write_text('extern "C" float docs_scalar(float);\n'
                      'namespace dm::nn { struct Tensor { float value; '
                      'template<class T> T item() const { return T(value); } }; '
                      'inline Tensor scalar(float x) { return {docs_scalar(x)}; } }\n')
    libdir = tmp_path / "lib"
    libdir.mkdir()
    library = libdir / ("libdatamunge.dylib" if sys.platform == "darwin" else "libdatamunge.so")
    subprocess.run([compiler, "-x", "c++", "-shared", "-fPIC", "-o", str(library), "-"],
                   input='extern "C" float docs_scalar(float x) { return x; }\n',
                   text=True, capture_output=True, check=True)
    pc = tmp_path / "pkgconfig"
    pc.mkdir()
    (pc / "datamunge.pc").write_text(
        f'Name: datamunge\nDescription: docs fixture\nVersion: 1\n'
        f'Cflags: -I{tmp_path / "include"}\nLibs: -L{libdir} -ldatamunge\nlibdir={libdir}\n')
    monkeypatch.setenv("PKG_CONFIG_PATH", str(pc))
    monkeypatch.setenv("PKG_CONFIG_LIBDIR", str(pc))
    monkeypatch.setenv("CXX", shlex.join([compiler, "-Werror"]))
    return header, library, pc


def test_explicit_package_and_multiword_compiler_compile_link_and_run(package, tmp_path, capsys):
    reports.validate_cpp_toolchain([document(tmp_path / "report.mepml")])
    assert "headers, linking and execution OK" in capsys.readouterr().out


@pytest.mark.parametrize("missing", ["header", "library"])
def test_broken_installation_reports_compile_or_link_failure(package, tmp_path, missing):
    header, library, _ = package
    (header if missing == "header" else library).unlink()
    with pytest.raises(reports.ToolchainError, match="compiling/linking <dm/dm.hpp>"):
        reports.validate_cpp_toolchain([document(tmp_path / "report.mepml")])


def test_native_compiler_diagnostic_is_preserved(package, tmp_path):
    package[0].write_text('#error broken installed header\n')
    with pytest.raises(reports.ToolchainError, match="broken installed header"):
        reports.validate_cpp_toolchain([document(tmp_path / "report.mepml")])


def test_execution_failure_is_not_reported_as_a_successful_link(package, tmp_path):
    header = package[0]
    header.write_text(header.read_text().replace("docs_scalar(x)", "docs_scalar(x) - 1"))
    with pytest.raises(reports.ToolchainError, match="executing the linked datamunge canary"):
        reports.validate_cpp_toolchain([document(tmp_path / "report.mepml")])


@pytest.mark.parametrize("no_pch", [False, True])
def test_missing_package_fails_before_pch_workers_and_status_changes(tmp_path, monkeypatch, no_pch):
    pytest.importorskip("markdown_it", reason="website integration uses the docs shell dependencies")
    monkeypatch.setitem(sys.modules, "reports", reports)
    spec = importlib.util.spec_from_file_location("website_build_check", ROOT / "scripts/website/build.py")
    website = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(website)
    source = document(tmp_path / "report.mepml")
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    status = status_dir / "old.json"
    status.write_text('{"files": [{"failed": 0}]}\n')
    original = status.read_bytes()
    empty = tmp_path / "empty-package-dir"
    empty.mkdir()
    monkeypatch.setenv("PKG_CONFIG_PATH", str(empty))
    monkeypatch.setenv("PKG_CONFIG_LIBDIR", str(empty))

    def unexpected(*args, **kwargs):
        raise AssertionError("PCH/report execution started before setup was checked")

    monkeypatch.setattr(reports, "toolchain", unexpected)
    monkeypatch.setattr(reports, "run_reports", unexpected)
    site = website.Site.__new__(website.Site)
    site.args = SimpleNamespace(reports=[source], no_reports=False, run=True,
                                status_dir=str(status_dir), force=True, no_pch=no_pch)
    site.out = str(tmp_path / "site")
    with pytest.raises(reports.ToolchainError, match="pkg-config datamunge discovery"):
        site.build_reports()
    assert list(status_dir.iterdir()) == [status]
    assert status.read_bytes() == original


def test_nonlibrary_blocks_do_not_require_an_installed_package(tmp_path, monkeypatch):
    source = tmp_path / "report.mepml"
    source.write_text('```bash\necho "#include <dm/dm.hpp>"\n```\n'
                      '```cpp\n#include <iostream>\nint main() {}\n```\n')

    def unexpected(*args, **kwargs):
        raise AssertionError("nonlibrary report checked the C++ library")

    monkeypatch.setattr(reports.subprocess, "run", unexpected)
    reports.validate_cpp_toolchain([str(source)])


def test_report_data_preparation_is_native_and_independent_of_checkout_scripts():
    reports = list((ROOT / "docs/mepml/examples").rglob("*.mepml"))
    native = []
    for path in reports:
        text = path.read_text()
        assert "scripts/prepare_" not in text, path
        # NN reports perform model/data work in C++; other collections may
        # also demonstrate native Python/R bindings after native preparation.
        if path.is_relative_to(ROOT / "docs/mepml/examples/nn"):
            assert not re.search(r"^```\{?python", text, re.M), path
        for block in re.findall(r'^```[^\n]*\n(.*?)^```', text, re.M | re.S):
            assert not re.search(r"python(?:3(?:\.\d+)?)? .*prepare_", block), path
            assert "pip install" not in block, path
            if "dm::datasets::prepare_example_data(" in block:
                assert "#include <dm/dm.hpp>" in block, path
                native.append(path)
    assert len(set(native)) >= 26


def test_report_prerequisites_expand_once_and_reject_cycles(tmp_path):
    parent, child = tmp_path / "parent.mepml", tmp_path / "child.mepml"
    parent.write_text("//? Title: Parent\n")
    child.write_text("//? Requires: parent.mepml\n")
    status = lambda path: str(tmp_path / (Path(path).stem + ".json"))
    result = reports.with_prerequisites([(str(child), status(child)), (str(parent), status(parent))], status)
    assert [Path(path).name for path, _ in result] == ["parent.mepml", "child.mepml"]
    parent.write_text("//? Requires: child.mepml\n")
    with pytest.raises(reports.ToolchainError, match="Cyclic"):
        reports.with_prerequisites([(str(child), status(child))], status)
    parent.write_text("//? Requires: missing.mepml\n")
    with pytest.raises(reports.ToolchainError, match="Missing report prerequisite"):
        reports.with_prerequisites([(str(parent), status(parent))], status)


@pytest.mark.parametrize("parent_fails", [False, True])
def test_parallel_reports_wait_for_native_checkpoint_producer(tmp_path, monkeypatch, parent_fails):
    parent, child = tmp_path / "parent.mepml", tmp_path / "child.mepml"
    parent.write_text("//? Title: Parent\n")
    child.write_text("//? Requires: parent.mepml\n")
    calls = []

    def run(executable, path, status, timeout, env):
        name = Path(path).stem
        if name == "child":
            assert calls == ["parent"]
        calls.append(name)
        Path(status).write_text(json.dumps({"files": [{"file": path, "failed": int(parent_fails),
                                                      "skipped": 0, "blocks": []}]}))
        return 0, ""

    monkeypatch.setattr(reports, "run_report", run)
    jobs = [(str(path), str(tmp_path / (path.stem + ".json"))) for path in [child, parent]]
    reports.run_reports("fixture", jobs, jobs=2)
    assert calls == (["parent"] if parent_fails else ["parent", "child"])
    if parent_fails:
        assert "required report failed" in reports.load_status(jobs[0][1])["error"]


def test_sibling_reports_cannot_capture_each_others_newest_figure(tmp_path, monkeypatch):
    """Shared working directories serialize while independent directories overlap."""
    import threading
    import time
    first_dir, other_dir = tmp_path / "first", tmp_path / "other"
    first_dir.mkdir()
    other_dir.mkdir()
    paths = [first_dir / "a.mepml", first_dir / "b.mepml", other_dir / "c.mepml"]
    for path in paths:
        path.write_text("//? Title: Fixture\n")
    overlap = threading.Barrier(2)
    captured = {}

    def run(executable, path, status, timeout, env):
        path = Path(path)
        newest = path.parent / "newest.svg"
        newest.write_text(path.stem)
        if path.stem in ("a", "c"):
            overlap.wait(timeout=5)
        time.sleep(.03)
        captured[path.stem] = newest.read_text()
        Path(status).write_text(json.dumps({"files": [{"file": str(path), "failed": 0,
                                                       "skipped": 0, "blocks": []}]}))
        return 0, ""

    monkeypatch.setattr(reports, "run_report", run)
    jobs = [(str(path), str(tmp_path / (path.stem + ".json"))) for path in paths]
    reports.run_reports("fixture", jobs, jobs=3)
    assert captured == {"a": "a", "b": "b", "c": "c"}


def test_public_dataset_reads_acquire_missing_sources_natively():
    checked = 0
    for path in (ROOT / "docs/mepml/examples").rglob("*.mepml"):
        for block in re.findall(r'^```[^\n]*\n(.*?)^```', path.read_text(), re.M | re.S):
            if "LoadSaliency(cache" in block:
                assert 'prepare_example_data("mnist-assets", cache.string())' in block, path
            for dataset in ("mnist", "cifar10", "cifar100"):
                if re.search(r'cache\s*/\s*"' + dataset + r'(?:/|\")', block) and re.search(
                        r'(?:load_mnist|load_cifar|LoadCifar|ReadIdx)', block):
                    assert 'dm::datasets::fetch("' + dataset + '"' in block, path
                    checked += 1
    assert checked >= 163


def test_missing_local_audio_sources_make_a_successful_report_stale(tmp_path):
    path=tmp_path/'wave.mepml';path.write_text(
        '<audio controls src="figures/cold.wav"></audio>\n'
        '<audio controls><source src="figures/prompt&amp;voice.wav" type="audio/wav"></audio>\n'
        '<audio src="https://example.org/remote.wav"></audio>\n')
    status=tmp_path/'status.json';status.write_text(json.dumps({'files':[{'failed':0}]}))
    os.utime(path,(10,10));os.utime(status,(20,20))
    assert reports.missing_figures(path)==['figures/cold.wav','figures/prompt&voice.wav']
    assert reports.is_stale(path,status)
    (tmp_path/'figures').mkdir();(tmp_path/'figures/cold.wav').write_bytes(b'cold')
    (tmp_path/'figures/prompt&voice.wav').write_bytes(b'prompt')
    assert not reports.missing_figures(path) and not reports.is_stale(path,status)
    (tmp_path/'figures/cold.wav').unlink()
    assert reports.is_stale(path,status)


def test_website_export_copies_both_direct_and_nested_audio_sources(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();(source/'figures').mkdir()
    for name,data in [('cold.wav',b'cold'),('prompt.wav',b'prompt')]:
        (source/'figures'/name).write_bytes(data)
    path=source/'wave.mepml';path.write_text('Fixture')
    page='<body><audio controls src="figures/cold.wav"></audio><audio controls><source src="figures/prompt.wav" type="audio/wav"></audio></body>'
    def convert(command,**kwargs):
        Path(command[-1]).write_text(page)
    monkeypatch.setattr(reports.subprocess,'run',convert)
    report=reports.Report(str(path),'nn/wave.mepml');out=tmp_path/'site'
    reports.export('fixture',report,out,lambda code,lang:code)
    for name,data in [('cold.wav',b'cold'),('prompt.wav',b'prompt')]:
        assert (out/'reports/nn/figures'/name).read_bytes()==data
    assert report.body.count('<audio')==2 and '<source src="figures/prompt.wav"' in report.body
