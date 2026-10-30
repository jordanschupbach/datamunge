"""Wide architecture SVGs retain native size through the real report exporter."""
import html
import importlib.util
from html.parser import HTMLParser
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "website_reports", Path(__file__).parents[2] / "scripts/website/reports.py")
reports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reports)


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def export_fixture(tmp_path, monkeypatch, name, marker):
    source = tmp_path / "source"
    source.mkdir()
    diagram = source / name
    diagram.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="500" '
                       + marker + '></svg>')
    document = source / "model.mepml"
    document.write_text("Source fixture")
    page = ('<body><figure><img src="' + html.escape(name)
            + '" alt="Exact model shape"></figure>'
              '<img src="https://example.test/remote.svg" alt="Remote"></body>')

    def convert(command, **kwargs):
        Path(command[-1]).write_text(page)

    monkeypatch.setattr(reports.subprocess, "run", convert)
    report = reports.Report(str(document), "nn/model.mepml")
    output = tmp_path / "site"
    reports.export("mep-mepml", report, str(output), lambda code, lang: code)
    parser = Tags()
    parser.feed(report.body)
    return report, parser.tags, output


def test_model_diagram_is_scrollable_and_keyboard_accessible(tmp_path, monkeypatch):
    report, tags, output = export_fixture(
        tmp_path, monkeypatch, "architecture.svg", 'data-model-layout="left-to-right"')
    wrappers = [attrs for tag, attrs in tags if tag == "span"]
    assert wrappers == [{"class": "model-diagram", "tabindex": "0", "role": "region",
                         "aria-label": "Network diagram; scroll to explore"}]
    images = [attrs for tag, attrs in tags if tag == "img"]
    assert images[0]["alt"] == "Exact model shape"
    assert images[1]["src"] == "https://example.test/remote.svg"
    assert (output / "reports/nn/architecture.svg").is_file()


def test_ordinary_svg_remains_responsive_without_a_wrapper(tmp_path, monkeypatch):
    report, tags, output = export_fixture(tmp_path, monkeypatch, "scatter.svg", "")
    assert not any(tag == "span" for tag, attrs in tags)
    assert '<img src="scatter.svg" alt="Exact model shape">' in report.body
    assert (output / "reports/nn/scatter.svg").is_file()


def test_escaped_local_filename_still_matches_diagram_metadata(tmp_path, monkeypatch):
    report, tags, output = export_fixture(
        tmp_path, monkeypatch, "model&layers.svg", 'data-model-layout="left-to-right"')
    assert sum(tag == "span" for tag, attrs in tags) == 1
    assert (output / "reports/nn/model&layers.svg").is_file()
    assert 'src="model&amp;layers.svg"' in report.body
