#!/usr/bin/env python3
"""Generate editable report navigation directly from the current source tree."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import re
from urllib.parse import quote

DESTINATION = Path('docs/mepml/overview/index.mepml')
COLLECTIONS = (Path('docs/mepml'),)


def inventory(root):
    """Discover current sources without conversion ledgers or historical catalogues."""
    root = Path(root).resolve()
    rows = []
    for directory, kind in ((COLLECTIONS[0], 'Documentation source'),
                            (Path('examples/mepml'), 'Legacy source')):
        for path in sorted((root / directory).rglob('*.mepml')):
            source = path.relative_to(root)
            if source in (DESTINATION, Path('examples/mepml/index.mepml')):
                continue
            relative = source.relative_to(directory)
            title = re.search(r'^//\? Title:\s*(.+)$', path.read_text(), re.M)
            rows.append(dict(source=source.as_posix(), stem=path.stem,
                             title=title[1].strip() if title else path.stem.replace('_', ' '),
                             kind=kind, group=relative.parent.as_posix()
                             if relative.parent != Path('.') else 'general'))
    return sorted(rows, key=lambda r: (r['kind'], r['group'].casefold(),
                                      r['title'].casefold(), r['source']))


def label(text):
    """Escape a source title for a native mepml link."""
    return re.sub(r'([\\\[\]|])', r'\\\1', ' '.join(text.split()))


def render(root):
    """Render navigation from live files; no report is executed."""
    root = Path(root).resolve()
    rows = inventory(root)
    text = '''//? Title: datamunge Report Catalogue
//? Subtitle: Worked examples by subject
//? Exports: both
//? Lang: en-GB

\\abstract(
Worked examples live in `docs/mepml/`, organized by subject and topic.
This navigation follows the source files currently present in the repository.
The website's Reports and build-status pages show generated pages and execution outcomes.
)
\\toc

> Using this catalogue

Open a title to visit its editable mepml source. The documentation website maps
these links to generated report pages. Refresh navigation with
`just docs-report-catalogue`; the website refreshes it before report discovery.

'''
    text += f'Current inventory: {len(rows)} sources, excluding this navigation page.\n\n'
    groups = defaultdict(list)
    for row in rows:
        groups[row['kind'], row['group']].append(row)
    for (kind, group), items in groups.items():
        text += f'> {label(kind)}: {label(group)}\n\n'
        for row in items:
            href = quote(os.path.relpath(root / row['source'], (root / DESTINATION).parent)
                         .replace(os.sep, '/'), safe='/')
            text += f"- [{label(row['title'])}|{href}] — `{row['source']}`.\n"
        text += '\n'
    return text, rows


def write(root):
    """Preserve the timestamp when generated navigation is unchanged."""
    root = Path(root).resolve()
    text, rows = render(root)
    path = root / DESTINATION
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return rows


def main():
    """Generate or check navigation against current filesystem discovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text, rows = render(args.root)
    if args.check:
        path = args.root / DESTINATION
        if not path.exists() or path.read_text() != text:
            parser.exit(1, 'Report catalogue is stale; run just docs-report-catalogue\n')
    else:
        write(args.root)
    print(f'{len(rows)} live sources')


if __name__ == '__main__':
    main()
