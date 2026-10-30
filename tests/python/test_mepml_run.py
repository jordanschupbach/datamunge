"""Report artifacts must be produced in document order even with parallel compilation."""
import importlib.util
from pathlib import Path
import time

SPEC = importlib.util.spec_from_file_location("mepml_run", Path(__file__).parents[2] / "scripts/mepml_run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_parallel_preparation_preserves_execution_order(tmp_path, monkeypatch):
    doc = tmp_path / "ordered.mepml"
    doc.write_text('''```{python}
from pathlib import Path
Path("state").write_text("first")
```
```{python}
from pathlib import Path
assert Path("state").read_text() == "first"
print("consumed earlier artifact")
```
''')
    original = runner.prepare_block
    def delayed(directory, language, options, code):
        if 'write_text("first")' in "\n".join(code):
            time.sleep(.15)
        return original(directory, language, options, code)
    monkeypatch.setattr(runner, "prepare_block", delayed)
    assert runner.run_document(str(doc), 2)
    assert "// consumed earlier artifact" in doc.read_text()


def test_failure_leaves_report_unchanged_and_cleans_prepared_blocks(tmp_path):
    doc = tmp_path / "failed.mepml"
    text = '```{python}\nraise RuntimeError("expected")\n```\n```{python}\nprint("later")\n```\n'
    doc.write_text(text)
    assert not runner.run_document(str(doc), 2)
    assert doc.read_text() == text
