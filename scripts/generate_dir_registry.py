#!/usr/bin/env python3
"""Generate include/datamunge/dir_registry.hpp: the member tables behind dm::dir().

C++ has no runtime reflection, so dm::dir(obj) answers from a table generated
ahead of time. The source of truth is the Doxygen XML that `just docs-audit`
already builds (EXTRACT_ALL is on, so every class is in it): for each public
class/struct under the root namespace (skipping `detail` namespaces), every
public member -- inherited ones included, like Python's dir() -- with its
signature and one-line description.

Usage:
    python3 scripts/generate_dir_registry.py [--xml build/debug/docs/audit/xml]
                                             [--out include/datamunge/dir_registry.hpp]

Run through `just generate-dir`, which refreshes the XML first. The output is
deterministic (sorted), so regenerating without header changes is a no-op.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT_NAMESPACE = "datamunge"
MAX_BRIEF = 90

# One-letter member kinds stored in the table (decoded by dir.hpp).
CONSTRUCTOR, METHOD, STATIC_METHOD, FIELD, STATIC_FIELD, TYPE = "c", "m", "s", "f", "F", "t"


def text_of(node: ET.Element | None) -> str:
    """All text under an XML node, whitespace-collapsed."""
    if node is None:
        return ""
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def tidy(code: str) -> str:
    """Normalises Doxygen's spacing in types/signatures: `std::vector< double > &` -> `std::vector<double>&`."""
    code = re.sub(r"\[\[[^\]]*\]\]\s*", "", code)  # [[nodiscard]] and friends
    code = re.sub(r"<\s+", "<", code)
    code = re.sub(r"\s+>", ">", code)
    code = re.sub(r"\s+([&*]+)", r"\1", code)
    code = re.sub(r"([&*]+)(?=[A-Za-z_])", r"\1 ", code)  # `T&name` -> `T& name`
    code = re.sub(r"\s+,", ",", code)
    code = re.sub(r"\(\s+", "(", code)
    code = re.sub(r"\s+\)", ")", code)
    return re.sub(r"\s+", " ", code).strip()


def first_sentence(text: str) -> str:
    """The first sentence of a description, cut to MAX_BRIEF characters."""
    match = re.match(r"(.+?[.!?])(\s|$)", text)
    sentence = match.group(1) if match else text
    if len(sentence) > MAX_BRIEF:
        sentence = sentence[: MAX_BRIEF - 3].rstrip() + "..."
    return sentence


def describe(node: ET.Element) -> str:
    """The brief description, falling back to the first sentence of the detailed one."""
    brief = text_of(node.find("briefdescription"))
    if not brief:
        detailed = node.find("detaileddescription")
        if detailed is not None:
            # Skip parameter/return lists and code: the leading paragraph only.
            para = detailed.find("para")
            brief = text_of(para) if para is not None else ""
            brief = re.split(r"\s(Parameters|Returns|Exceptions|Example)\b", brief)[0]
    return first_sentence(brief)


def is_public_api(name: str) -> bool:
    """A class worth listing: in the root namespace, not in a detail/internal namespace."""
    parts = name.split("::")
    if parts[0] != ROOT_NAMESPACE:
        return False
    return not any("detail" in part or part.startswith("_") or part.startswith("@") for part in parts)


def short_name(qualified: str) -> str:
    """The class's own name without namespaces or template arguments."""
    return re.sub(r"<.*>$", "", qualified.split("::")[-1])


def entry_for(member: ET.Element, class_short: str) -> tuple[str, str, str, str] | None:
    """(kind, name, signature, brief) for one public member, or None for members dir() skips."""
    kind = member.get("kind")
    name = member.findtext("name") or ""
    static = member.get("static") == "yes"
    brief = describe(member)
    if kind == "function":
        if name.startswith("~") or member.get("prot") != "public":
            return None
        args = tidy(member.findtext("argsstring") or "")
        result = tidy(text_of(member.find("type")))
        if name == class_short:
            return (CONSTRUCTOR, name, f"{name}{args}", brief)
        signature = f"{name}{args}" + (f" -> {result}" if result else "")
        return (STATIC_METHOD if static else METHOD, name, signature, brief)
    if kind == "variable":
        result = tidy(text_of(member.find("type")))
        return (STATIC_FIELD if static else FIELD, name, f"{result} {name}".strip(), brief)
    if kind in ("typedef", "enum"):
        definition = tidy(member.findtext("definition") or f"{kind} {name}")
        return (TYPE, name, definition.replace("typedef ", "using ", 1) if kind == "typedef" else f"enum {name}", brief)
    return None


def collect(xml_dir: Path) -> dict[str, tuple[str, list[tuple[str, str, str, str]]]]:
    """Maps each listed class's qualified name to (class brief, members)."""
    members_by_id: dict[str, ET.Element] = {}
    compounds: list[ET.Element] = []
    for path in sorted(xml_dir.glob("*.xml")):
        if not (path.name.startswith("class") or path.name.startswith("struct")):
            continue
        compound = ET.parse(path).getroot().find("compounddef")
        if compound is None:
            continue
        compounds.append(compound)
        for member in compound.iter("memberdef"):
            members_by_id[member.get("id", "")] = member

    table: dict[str, tuple[str, list[tuple[str, str, str, str]]]] = {}
    for compound in compounds:
        name = compound.findtext("compoundname") or ""
        if not is_public_api(name) or compound.get("prot", "public") != "public":
            continue
        class_short = short_name(name)
        entries: dict[tuple[str, str], tuple[str, str, str, str]] = {}
        listing = compound.find("listofallmembers")
        refs = listing.findall("member") if listing is not None else []
        for ref in refs:
            if ref.get("prot") != "public":
                continue
            member = members_by_id.get(ref.get("refid", ""))
            if member is None:
                continue
            entry = entry_for(member, class_short)
            if entry is None:
                continue
            # An inherited constructor is not a constructor of this class.
            if entry[0] == CONSTRUCTOR and short_name(text_of(ref.find("scope"))) != class_short:
                continue
            entries.setdefault((entry[0], entry[2]), entry)
        table[name] = (describe(compound), sorted(entries.values(), key=lambda e: (e[1], e[2])))
    return table


def cpp_raw(text: str) -> str:
    """A C++ raw string literal holding `text`."""
    if ")dm\"" in text:
        raise ValueError(f"cannot embed {text!r}")
    return f'R"dm({text})dm"'


def render(table: dict[str, tuple[str, list[tuple[str, str, str, str]]]]) -> str:
    """The generated header's text."""
    lines = [
        "#pragma once",
        "",
        "/// @file dir_registry.hpp",
        "/// @brief GENERATED by scripts/generate_dir_registry.py (`just generate-dir`); do not edit.",
        "///",
        "/// The member tables behind dm::dir(): one entry per public class, sorted by",
        "/// qualified name (template arguments dropped). Each entry's text is the class's",
        "/// one-line description, then one member per line as `kind<TAB>name<TAB>signature<TAB>brief`.",
        "",
        "#include <cstddef>",
        "#include <string_view>",
        "",
        f"/// @brief Core library APIs and their algorithm, data, and numerical submodules.",
        f"namespace {ROOT_NAMESPACE}::dir_detail {{",
        "",
        "/// @brief One class's generated member table.",
        "struct RegistryEntry {",
        "  std::string_view type;     ///< Qualified class name without template arguments.",
        "  std::string_view members;  ///< Description line, then `kind\\tname\\tsignature\\tbrief` lines.",
        "};",
        "",
        "/// @brief Every listed class, sorted by `type` for binary search.",
        "inline constexpr RegistryEntry kRegistry[] = {",
    ]
    for name in sorted(table):
        brief, members = table[name]
        body = "\n".join([brief] + ["\t".join(m).replace("\n", " ") for m in members])
        # Explicit byte lengths: constructing a string_view from a bare literal makes the compiler
        # count every character in constant evaluation, which overruns clang's step limit here.
        lines.append(
            f"    {{std::string_view({cpp_raw(name)}, {len(name.encode())}), "
            f"std::string_view({cpp_raw(body)}, {len(body.encode())})}},"
        )
    lines += [
        "};",
        "",
        "/// @brief Number of classes in kRegistry.",
        "inline constexpr std::size_t kRegistrySize = sizeof(kRegistry) / sizeof(kRegistry[0]);",
        "",
        f"}}  // namespace {ROOT_NAMESPACE}::dir_detail",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml", type=Path, default=Path("build/debug/docs/audit/xml"))
    parser.add_argument("--out", type=Path, default=Path(f"include/{ROOT_NAMESPACE}/dir_registry.hpp"))
    args = parser.parse_args()
    if not (args.xml / "index.xml").exists():
        print(f"no Doxygen XML in {args.xml}; run `just docs-audit` first", file=sys.stderr)
        return 1
    table = collect(args.xml)
    text = render(table)
    if args.out.exists() and args.out.read_text() == text:
        print(f"{args.out}: up to date ({len(table)} classes)")
        return 0
    args.out.write_text(text)
    members = sum(len(m) for _, m in table.values())
    print(f"{args.out}: {len(table)} classes, {members} members, {len(text) // 1024} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
