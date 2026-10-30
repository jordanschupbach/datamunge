#!/usr/bin/env python3
"""Audit maintained C/C++ symbols using a complete Doxygen XML inventory.

Run ``just docs-audit`` to regenerate XML before checking coverage. This checks
the presence of descriptions, not their semantic quality, parameter completeness,
or function-local statics (which Doxygen does not enumerate).
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

## Repository checkout containing the canonical sources and this audit script.
ROOT = Path(__file__).resolve().parents[1]
## Verified source batches enforced by CI while the remaining backlog is documented.
COMPLETED_PATHS = ROOT / "docs/documented-paths.txt"
## Source roots covered by the initial core audit; generated bindings live elsewhere.
SOURCE_ROOTS = ("include", "src/datamunge")
## Keep these extensions synchronized with the core FILE_PATTERNS in docs/Doxyfile.
SOURCE_SUFFIXES = frozenset({".hpp", ".h", ".cpp", ".c", ".inl"})
## Doxygen compound categories that represent source definitions with descriptions.
COMPOUND_KINDS = frozenset(
    {"file", "namespace", "class", "struct", "union", "interface", "protocol", "category", "exception"}
)
## Extraction switches needed to prevent private/internal definitions disappearing from XML.
REQUIRED_EXTRACTION = (
    "EXTRACT_ALL", "EXTRACT_PRIVATE", "EXTRACT_STATIC", "EXTRACT_LOCAL_CLASSES",
    "EXTRACT_ANON_NSPACES", "INTERNAL_DOCS",
)


@dataclass(frozen=True)
class Symbol:
    """One Doxygen definition, deduplicated across file, group, and namespace pages."""

    ## Doxygen's unique identifier, including an overload-specific signature hash.
    id: str
    ## Doxygen definition category, such as function, variable, enum, or enumvalue.
    kind: str
    ## Qualified name when available; function names also include their arguments.
    name: str
    ## POSIX source path relative to the checkout root.
    path: str
    ## One-based declaration line, or the enclosing enum's line for its values.
    line: int
    ## Whether Doxygen resolved any brief, detailed, or in-body description.
    documented: bool


def source_path(value: str, root: Path) -> str | None:
    """Return a maintained checkout-relative source path, excluding external definitions."""
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None
    if path.suffix in SOURCE_SUFFIXES and any(
        relative.startswith(prefix + "/") for prefix in SOURCE_ROOTS
    ):
        return relative
    return None


def has_description(element: ET.Element) -> bool:
    """Check resolved prose, including copydoc, rather than counting the name or signature."""
    return any(
        "".join(description.itertext()).strip()
        for tag in ("briefdescription", "detaileddescription", "inbodydescription")
        if (description := element.find(tag)) is not None
    )


def read_inventory(xml_dir: Path, root: Path = ROOT) -> list[Symbol]:
    """Read definitions and enum values from every compound referenced by index.xml.

    Missing XML, an empty inventory, omitted source files, and sources newer than
    the index raise ValueError, so an incomplete/stale run cannot pass coverage.
    Shared definitions are counted once by their Doxygen ID.
    """
    index_path = xml_dir / "index.xml"
    index = ET.parse(index_path).getroot()
    config = ET.parse(xml_dir / "Doxyfile.xml").getroot()
    for option in REQUIRED_EXTRACTION:
        if config.findtext(f"./option[@id='{option}']/value") != "YES":
            raise ValueError(f"{option} must be YES for a complete coverage inventory")
    if any(value.text for value in config.findall("./option[@id='EXCLUDE_SYMBOLS']/value")):
        raise ValueError("EXCLUDE_SYMBOLS must be empty for a complete coverage inventory")
    symbols: dict[str, Symbol] = {}
    files: set[str] = set()
    for entry in index.findall("compound"):
        compound = ET.parse(xml_dir / (entry.attrib["refid"] + ".xml")).getroot().find("compounddef")
        if compound is None:
            raise ValueError(f"Missing compound definition for {entry.attrib['refid']}")
        definitions = [compound] if compound.attrib["kind"] in COMPOUND_KINDS else []
        definitions.extend(compound.findall("./sectiondef/memberdef"))
        for definition in definitions:
            location = definition.find("location")
            if location is None:
                continue
            path = source_path(location.get("file", ""), root)
            if path is None:
                continue
            kind = definition.attrib["kind"]
            if kind == "file":
                files.add(path)
            name = (
                definition.findtext("qualifiedname")
                or definition.findtext("compoundname")
                or definition.findtext("name")
                or path
            )
            if kind in {"function", "friend", "signal", "slot"}:
                name += definition.findtext("argsstring", "")
            symbol = Symbol(
                id=definition.attrib["id"],
                kind=kind,
                name=name,
                path=path,
                line=int(location.get("line", "1")),
                documented=has_description(definition),
            )
            previous = symbols.get(symbol.id)
            if previous is None or symbol.documented:
                symbols[symbol.id] = symbol
            for value in definition.findall("enumvalue"):
                enum_value = Symbol(
                    id=value.attrib["id"],
                    kind="enumvalue",
                    name=f"{name}::{value.findtext('name')}",
                    path=path,
                    line=symbol.line,
                    documented=has_description(value),
                )
                previous = symbols.get(enum_value.id)
                if previous is None or enum_value.documented:
                    symbols[enum_value.id] = enum_value
    expected = {
        path.relative_to(root).as_posix()
        for prefix in SOURCE_ROOTS
        for path in (root / prefix).rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    }
    if not expected or not symbols:
        raise ValueError("Empty source or symbol inventory; coverage cannot be established")
    if omitted := expected - files:
        raise ValueError("Source files absent from XML; regenerate Doxygen: " + ", ".join(sorted(omitted)))
    if removed := files - expected:
        raise ValueError("Removed files still present in XML; regenerate Doxygen: " + ", ".join(sorted(removed)))
    if newer := sorted(path for path in expected if (root / path).stat().st_mtime_ns > index_path.stat().st_mtime_ns):
        raise ValueError("Sources newer than XML; regenerate Doxygen: " + ", ".join(newer))
    for relative in ("docs/Doxyfile", "docs/CMakeLists.txt"):
        config_source = root / relative
        if config_source.exists() and config_source.stat().st_mtime_ns > index_path.stat().st_mtime_ns:
            raise ValueError(f"{relative} is newer than XML; reconfigure and regenerate Doxygen")
    return sorted(symbols.values(), key=lambda symbol: (symbol.path, symbol.line, symbol.kind, symbol.name))


def in_scope(symbol: Symbol, prefix: str) -> bool:
    """Match an exact source path or its descendants, without matching sibling prefixes."""
    return symbol.path == prefix or symbol.path.startswith(prefix.rstrip("/") + "/")


def summarize(symbols: list[Symbol]) -> dict:
    """Return deterministic total, category, and source-area documentation counts."""
    def counts(items: list[Symbol]) -> dict:
        """Count documented and missing definitions in one reporting bucket."""
        documented = sum(symbol.documented for symbol in items)
        return {"total": len(items), "documented": documented, "missing": len(items) - documented}

    def area(symbol: Symbol) -> str:
        """Group core definitions by their source directory rather than absolute checkout path."""
        parts = Path(symbol.path).parts
        return "/".join(parts[:3] if len(parts) > 3 else parts[:2])

    return {
        **counts(symbols),
        "by_kind": {
            kind: counts([symbol for symbol in symbols if symbol.kind == kind])
            for kind in sorted({symbol.kind for symbol in symbols})
        },
        "by_area": {
            name: counts([symbol for symbol in symbols if area(symbol) == name])
            for name in sorted({area(symbol) for symbol in symbols})
        },
    }


def write_json(path: Path, value: object) -> None:
    """Write a stable UTF-8 report, creating its parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Report missing definitions; return 1 for failed coverage or 2 for invalid inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=ROOT / "build/debug/docs/audit/xml")
    parser.add_argument("--path", action="append", default=[], help="Check only this repo-relative file/directory (repeatable)")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if any selected symbol lacks documentation")
    parser.add_argument("--completed", action="store_true", help="Select the verified paths in docs/documented-paths.txt")
    parser.add_argument("--json", type=Path, help="Write counts and a complete missing-symbol list")
    parser.add_argument("--summary", type=Path, help="Write only compact counts, suitable for a tracked baseline")
    parser.add_argument("--limit", type=int, default=20, help="Maximum missing symbols printed (default: 20)")
    args = parser.parse_args(argv)
    try:
        if args.completed:
            completed = [line.split("#", 1)[0].strip() for line in COMPLETED_PATHS.read_text().splitlines()]
            completed = [prefix for prefix in completed if prefix]
            if not completed:
                raise ValueError("The completed-path manifest is empty")
            args.path.extend(completed)
        symbols = read_inventory(args.xml)
        for prefix in args.path:
            if not any(in_scope(symbol, prefix) for symbol in symbols):
                raise ValueError(f"No maintained definitions matched --path {prefix!r}")
        if args.path:
            symbols = [symbol for symbol in symbols if any(in_scope(symbol, prefix) for prefix in args.path)]
    except (OSError, ET.ParseError, ValueError, KeyError) as error:
        print(f"Documentation inventory error: {error}", file=sys.stderr)
        return 2
    summary = summarize(symbols)
    missing = [symbol for symbol in symbols if not symbol.documented]
    print(f"Documented {summary['documented']}/{summary['total']} definitions; {len(missing)} missing.")
    for kind, count in sorted(Counter(symbol.kind for symbol in missing).items()):
        print(f"  {kind}: {count} missing")
    for symbol in missing[:max(0, args.limit)]:
        print(f"{symbol.path}:{symbol.line}: {symbol.kind} {symbol.name}")
    if len(missing) > args.limit:
        print("Use --json to save the complete missing-symbol list.")
    if args.json:
        write_json(args.json, {"summary": summary, "missing": [asdict(symbol) for symbol in missing]})
    if args.summary:
        write_json(args.summary, summary)
    return int(args.check and bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
