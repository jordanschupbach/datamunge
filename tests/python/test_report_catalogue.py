"""Catalogue membership, migration refresh and real source-link export behavior."""
import importlib.util
from html.parser import HTMLParser
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[2]

def load(name):
    spec = importlib.util.spec_from_file_location('catalogue_test_' + name, ROOT / 'scripts/website' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

catalogue, reports = load('catalogue'), load('reports')


def fixture(root):
    old = root / 'examples/mepml/old.mepml'
    old.parent.mkdir(parents=True)
    old.write_text('//? Title: Current legacy title\n')
    doc = root / 'docs/mepml/math/a & b.mepml'
    doc.parent.mkdir(parents=True)
    doc.write_text('//? Title: Current [title] | more\n')
    return old, doc


def test_catalogue_refreshes_a_migration_without_stale_links(tmp_path):
    old, doc = fixture(tmp_path)
    rows = catalogue.write(tmp_path)
    assert len(rows) == 2
    text = (tmp_path / catalogue.DESTINATION).read_text()
    assert 'a%20%26%20b.mepml' in text
    assert 'Legacy source' in text
    stamp = (tmp_path / catalogue.DESTINATION).stat().st_mtime_ns
    catalogue.write(tmp_path)
    assert (tmp_path / catalogue.DESTINATION).stat().st_mtime_ns == stamp
    new = tmp_path / 'docs/mepml/graph/old.mepml'
    new.parent.mkdir(parents=True)
    old.rename(new)
    rows = catalogue.write(tmp_path)
    assert next(r for r in rows if r['stem'] == 'old')['kind'] == 'Documentation source'
    assert 'examples/mepml/old.mepml' not in (tmp_path / catalogue.DESTINATION).read_text()
    new.unlink()
    assert len(catalogue.inventory(tmp_path)) == 1


def test_current_discovery_allows_equal_stems_in_different_topics(tmp_path):
    old, _ = fixture(tmp_path)
    another = tmp_path / 'docs/mepml/graph/old.mepml'
    another.parent.mkdir(parents=True, exist_ok=True)
    another.write_bytes(old.read_bytes())
    assert len([row for row in catalogue.inventory(tmp_path) if row['stem'] == 'old']) == 2


def test_merged_nn_collection_preserves_site_namespace(tmp_path):
    fixture(tmp_path)
    source = tmp_path / "docs/mepml/nn/3_generative_models/3.4_diffusion/ddpm.mepml"
    source.parent.mkdir(parents=True)
    source.write_text('//? Title: DDPM\n')
    assert str(source) in reports.discover_reports(tmp_path)
    assert reports.report_rel(source, reports.example_roots(tmp_path)) == "nn/3_generative_models/3.4_diffusion/ddpm.mepml"
    row = next(row for row in catalogue.inventory(tmp_path) if row['source'] == source.relative_to(tmp_path).as_posix())
    assert row['group'] == 'nn/3_generative_models/3.4_diffusion'
    from unittest.mock import patch
    with patch.object(reports, 'example_roots', return_value=[str(tmp_path / 'docs/mepml')] * 2):
        with pytest.raises(reports.ToolchainError, match="Duplicate report path"):
            reports.discover_reports(tmp_path)


def test_real_repository_covers_all_original_entries_and_current_sources():
    rows = catalogue.inventory(ROOT)
    expected = {str(p.relative_to(ROOT)) for p in (ROOT / 'docs/mepml').rglob('*.mepml')
                if p.relative_to(ROOT) != catalogue.DESTINATION}
    assert {row['source'] for row in rows} == expected
    assert all((ROOT / row['source']).is_file() for row in rows)
    assert len({row['source'] for row in rows}) == len(rows)
    assert not any(row['source'] == str(catalogue.DESTINATION) for row in rows)


def test_resolver_keeps_fragments_queries_remote_and_outside_paths(tmp_path):
    old, doc = fixture(tmp_path)
    source = tmp_path / catalogue.DESTINATION
    source.parent.mkdir(parents=True)
    page = 'reports/overview/index.html'
    known = {str(doc.resolve()): 'reports/math/a & b.html'}
    local = lambda p: reports.package_legacy_source(tmp_path, tmp_path / 'site', p)
    resolve = lambda h: reports.resolve_source_link(h, source, page, known, tmp_path, local)
    import os
    href = os.path.relpath(doc, source.parent).replace(' ', '%20').replace('&', '%26')
    assert resolve(href + '?q=1&n=2#method') == '../math/a%20%26%20b.html?q=1&n=2#method'
    oldhref = os.path.relpath(old, source.parent)
    assert resolve(oldhref) == '../sources/examples/mepml/old.mepml'
    assert (tmp_path / 'site/reports/sources/examples/mepml/old.mepml').read_bytes() == old.read_bytes()
    for href in ['https://example.test/x.mepml#part', '//example.test/x', '#same', '/absolute.mepml', '?query', '../figures/download.svg', '../../../../../../outside.mepml']:
        assert resolve(href) == href
    assert reports.package_legacy_source(tmp_path, tmp_path / 'site', '../outside.mepml') is None


def test_real_export_rewrites_anchors_and_copies_live_source(tmp_path, monkeypatch):
    old, doc = fixture(tmp_path)
    source = tmp_path / catalogue.DESTINATION
    source.parent.mkdir(parents=True)
    source.write_text('Navigation source')
    report = reports.Report(str(source), 'overview/index.mepml')
    import os
    document_href = os.path.relpath(doc, source.parent).replace(' ', '%20').replace('&', '%26')
    legacy_href = os.path.relpath(old, source.parent)
    page = f'<body><a href="{document_href}#part">Doc</a><a href="{legacy_href}">Legacy</a><a href="https://example.test">Remote</a></body>'
    def convert(command, **kwargs):
        Path(command[-1]).write_text(page)
    monkeypatch.setattr(reports.subprocess, 'run', convert)
    known = {str(doc.resolve()): 'reports/math/a & b.html'}
    out = tmp_path / 'site'
    link = lambda h: reports.resolve_source_link(h, source, report.url, known, tmp_path,
                                                lambda p: reports.package_legacy_source(tmp_path, out, p))
    reports.export('mep-mepml', report, str(out), lambda text, lang: text, link=link)
    assert 'href="../math/a%20%26%20b.html#part"' in report.body
    assert 'href="../sources/examples/mepml/old.mepml"' in report.body
    assert 'href="https://example.test"' in report.body
    assert (out / 'reports/sources/examples/mepml/old.mepml').read_bytes() == old.read_bytes()


def test_native_catalogue_links_render_as_real_anchors(tmp_path):
    import shutil
    import subprocess
    if not shutil.which('mep-mepml'):
        pytest.skip('mep-mepml is required for the actual export integration')
    fixture(tmp_path)
    catalogue.write(tmp_path)
    output = tmp_path / 'export'
    subprocess.run(['mep-mepml', 'build', '--no-post', '-d', str(output),
                    str(tmp_path / catalogue.DESTINATION), 'html'], check=True, capture_output=True)
    class Anchors(HTMLParser):
        def __init__(self):super().__init__();self.hrefs=[]
        def handle_starttag(self,tag,attrs):
            if tag=='a':self.hrefs.append(dict(attrs).get('href',''))
    parser=Anchors();parser.feed((output/'index.html').read_text())
    assert len([h for h in parser.hrefs if h.endswith('.mepml')]) == 2
    assert 'Current [title] | more</a>' in (output/'index.html').read_text()
