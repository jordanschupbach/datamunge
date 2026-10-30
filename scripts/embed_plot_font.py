#!/usr/bin/env python3
"""Generate src/datamunge/plot/cmu_serif_font_data.hpp from CMU Serif Roman (cmunrm.otf).

The plot engine draws text with Computer Modern (CMU Serif, SIL Open Font License 1.1).
This script subsets the font to the characters plots realistically need, then writes:

- glyph outlines (move/line/quad/cubic/close commands in font units) for the raster engine,
- advance widths and pair kerning (from GPOS 'kern'),
- the subset font as WOFF bytes, embedded by SVG output via @font-face.

Math symbols CMU Serif lacks (<= >= ~= != infinity, sums, ...) are taken from Latin Modern Math
(the same Computer Modern design, GUST Font License) for the raster outlines only; SVG viewers fall
back per character to an installed font for those.

The output is deterministic for a given input font. Regenerate with `just embed-plot-font`.
Requires fontTools (`python3Packages.fonttools` in nix).
"""

from __future__ import annotations

import argparse
import glob
import io
import sys
from pathlib import Path

from fontTools import subset
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "src" / "datamunge" / "plot" / "cmu_serif_font_data.hpp"

# Printable ASCII, Latin-1, Greek, and the symbols that show up in axis labels and annotations.
CODEPOINTS = (
    list(range(0x20, 0x7F))
    + list(range(0xA0, 0x100))
    + list(range(0x391, 0x3AA))
    + list(range(0x3B1, 0x3CA))
    + [0x3D1, 0x3D5, 0x3F5]  # theta, phi, epsilon variants
    + [0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2026, 0x2030, 0x2032, 0x2033]
    + [0x2070, 0x2071] + list(range(0x2074, 0x207F)) + list(range(0x2080, 0x208F))
    + [0x2190, 0x2191, 0x2192, 0x2193, 0x2194]
    + [0x2202, 0x2207, 0x2208, 0x220F, 0x2211, 0x2212, 0x221A, 0x221D, 0x221E, 0x222B]
    + [0x223C, 0x2248, 0x2260, 0x2261, 0x2264, 0x2265, 0x22C5]
)


def find_store_file(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern))
    return Path(hits[-1]) if hits else None


def find_default_font() -> Path | None:
    return find_store_file("/nix/store/*-cm-unicode-*/fonts/opentype/public/cm-unicode/cmunrm.otf")


def find_default_fallback() -> Path | None:
    return find_store_file("/nix/store/*-lm-math-*/fonts/opentype/public/lm-math/latinmodern-math.otf")


class OutlinePen(BasePen):
    """Records an outline as (op, points) with BasePen splitting multi-segment curves."""

    MOVE, LINE, QUAD, CUBIC, CLOSE = range(5)

    def __init__(self, glyph_set):
        super().__init__(glyph_set)
        self.ops: list[int] = []
        self.points: list[int] = []

    def _add(self, op, *pts):
        self.ops.append(op)
        for x, y in pts:
            self.points += [round(x), round(y)]

    def _moveTo(self, p):
        self._add(self.MOVE, p)

    def _lineTo(self, p):
        self._add(self.LINE, p)

    def _qCurveToOne(self, p1, p2):
        self._add(self.QUAD, p1, p2)

    def _curveToOne(self, p1, p2, p3):
        self._add(self.CUBIC, p1, p2, p3)

    def _closePath(self):
        self._add(self.CLOSE)

    def _endPath(self):
        self._add(self.CLOSE)


def subset_font(src: Path, flavor: str | None) -> TTFont:
    options = subset.Options()
    options.layout_features = ["kern"]
    options.name_IDs = ["*"]
    options.notdef_outline = True
    options.hinting = False
    options.desubroutinize = True
    options.flavor = flavor
    font = TTFont(str(src))
    subsetter = subset.Subsetter(options)
    subsetter.populate(unicodes=CODEPOINTS)
    subsetter.subset(font)
    return font


def kerning_pairs(font: TTFont) -> dict[tuple[str, str], int]:
    """Collects first-wins x-advance adjustments from every GPOS 'kern' PairPos lookup."""
    pairs: dict[tuple[str, str], int] = {}
    if "GPOS" not in font:
        return pairs
    gpos = font["GPOS"].table
    lookup_ids = set()
    for record in gpos.FeatureList.FeatureRecord:
        if record.FeatureTag == "kern":
            lookup_ids.update(record.Feature.LookupListIndex)
    for index in sorted(lookup_ids):
        lookup = gpos.LookupList.Lookup[index]
        for sub in lookup.SubTable:
            if lookup.LookupType == 9:  # extension wrapper around the real subtable
                if sub.ExtensionLookupType != 2:
                    continue
                sub = sub.ExtSubTable
            elif lookup.LookupType != 2:  # only pair adjustment carries kerning
                continue
            if sub.Format == 1:
                for first, pair_set in zip(sub.Coverage.glyphs, sub.PairSet):
                    for rec in pair_set.PairValueRecord:
                        value = getattr(rec.Value1, "XAdvance", 0) if rec.Value1 else 0
                        if value:
                            pairs.setdefault((first, rec.SecondGlyph), value)
            elif sub.Format == 2:
                class1 = sub.ClassDef1.classDefs
                class2 = sub.ClassDef2.classDefs
                seconds = font.getGlyphOrder()
                for first in sub.Coverage.glyphs:
                    row = sub.Class1Record[class1.get(first, 0)]
                    for second in seconds:
                        rec = row.Class2Record[class2.get(second, 0)]
                        value = getattr(rec.Value1, "XAdvance", 0) if rec.Value1 else 0
                        if value:
                            pairs.setdefault((first, second), value)
    return pairs


def chunked(values, per_line=24):
    for i in range(0, len(values), per_line):
        yield ", ".join(str(v) for v in values[i : i + per_line])


def generate(src: Path, fallback: Path | None) -> str:
    font = subset_font(src, None)
    upem = font["head"].unitsPerEm
    hhea = font["hhea"]

    # (font, glyph set, cmap, hmtx) per source; CMU Serif wins, the fallback fills gaps.
    sources = [(font, font.getGlyphSet(), font.getBestCmap(), font["hmtx"])]
    if fallback is not None:
        extra = TTFont(str(fallback))
        if extra["head"].unitsPerEm != upem:
            raise SystemExit("fallback font must share the primary font's unitsPerEm")
        sources.append((extra, extra.getGlyphSet(), extra.getBestCmap(), extra["hmtx"]))

    entries = []  # (codepoint, glyph name, advance, op_begin, op_count, pt_begin)
    ops: list[int] = []
    points: list[int] = []
    borrowed = 0
    for cp in sorted(set(CODEPOINTS) | set(sources[0][2])):
        for rank, (_font, glyph_set, cmap, hmtx) in enumerate(sources):
            if cp in cmap:
                break
        else:
            continue
        name = cmap[cp]
        borrowed += rank > 0
        pen = OutlinePen(glyph_set)
        glyph_set[name].draw(pen)
        # Fallback glyph names may collide with primary ones; only primary names matter for kerning.
        key = name if rank == 0 else f"fallback:{name}"
        entries.append((cp, key, hmtx[name][0], len(ops), len(pen.ops), len(points) // 2))
        ops += pen.ops
        points += pen.points

    index_of = {name: i for i, (_, name, *_rest) in enumerate(entries)}
    kerns = sorted(
        (index_of[a], index_of[b], v)
        for (a, b), v in kerning_pairs(font).items()
        if a in index_of and b in index_of
    )

    woff = io.BytesIO()
    subset_font(src, "woff").save(woff)
    woff_bytes = list(woff.getvalue())

    family = font["name"].getDebugName(1) or "CMU Serif"
    version = font["name"].getDebugName(5) or "unknown"

    out = io.StringIO()
    w = out.write
    w("// GENERATED by scripts/embed_plot_font.py -- do not edit by hand.\n")
    w(f"// Source: {src.name} ({family}, {version}).\n")
    if fallback is not None:
        w(f"// {borrowed} math symbols missing from it come from {fallback.name} (Latin Modern Math,\n")
        w("// GUST Font License).\n")
    w("// CMU Serif is (c) Andrey V. Panov, based on Donald Knuth's Computer Modern, and is\n")
    w("// licensed under the SIL Open Font License 1.1 (see resources/fonts/OFL-cm-unicode.txt).\n")
    w("// Private to src/datamunge/plot.cpp; not an installed header.\n")
    w("#pragma once\n\n#include <cstdint>\n\n")
    w("/// @brief Embedded Computer Modern (CMU Serif Roman) outlines, metrics, kerning and WOFF data.\n")
    w("namespace datamunge::graphics::detail::cmu_serif {\n\n")
    w(f"inline constexpr int units_per_em = {upem};\n")
    w(f"inline constexpr int ascender = {hhea.ascent};\n")
    w(f"inline constexpr int descender = {hhea.descent};\n\n")
    w("/// @brief Outline commands; MOVE/LINE take 1 point, QUAD 2, CUBIC 3, CLOSE 0.\n")
    w("enum Op : std::uint8_t { MOVE = 0, LINE = 1, QUAD = 2, CUBIC = 3, CLOSE = 4 };\n\n")
    w("/// @brief One glyph: its code point, advance (font units) and slices of ops/points.\n")
    w("struct GlyphEntry {\n  char32_t codepoint; ///< Unicode scalar value, table sorted by it\n")
    w("  std::int16_t advance; ///< horizontal advance in font units\n")
    w("  std::uint32_t op_begin; ///< first index into ops\n")
    w("  std::uint32_t op_count; ///< number of ops\n")
    w("  std::uint32_t point_begin; ///< first (x, y) pair index into points\n};\n\n")
    w("/// @brief Pair kerning between two glyph-table indices, in font units.\n")
    w("struct KernPair {\n  std::uint16_t left; ///< glyph index of the left character\n")
    w("  std::uint16_t right; ///< glyph index of the right character\n")
    w("  std::int16_t value; ///< x-advance adjustment\n};\n\n")
    w("inline constexpr GlyphEntry glyphs[] = {\n")
    for cp, _name, adv, ob, oc, pb in entries:
        w(f"  {{0x{cp:04X}, {adv}, {ob}, {oc}, {pb}}},\n")
    w("};\n\n")
    w("inline constexpr std::uint8_t ops[] = {\n")
    for line in chunked(ops, 40):
        w(f"  {line},\n")
    w("};\n\n")
    w("inline constexpr std::int16_t points[] = {\n")
    for line in chunked(points, 20):
        w(f"  {line},\n")
    w("};\n\n")
    w("inline constexpr KernPair kerning[] = {\n")
    for a, b, v in kerns:
        w(f"  {{{a}, {b}, {v}}},\n")
    if not kerns:
        w("  {0, 0, 0},\n")
    w("};\n\n")
    w("/// @brief The same subset as a WOFF file, for SVG @font-face embedding.\n")
    w("inline constexpr std::uint8_t woff[] = {\n")
    for line in chunked(woff_bytes, 24):
        w(f"  {line},\n")
    w("};\n\n} // namespace datamunge::graphics::detail::cmu_serif\n")
    print(
        f"glyphs={len(entries)} ops={len(ops)} points={len(points) // 2} "
        f"kern_pairs={len(kerns)} woff_bytes={len(woff_bytes)} borrowed={borrowed}",
        file=sys.stderr,
    )
    return out.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font", type=Path, default=None, help="path to cmunrm.otf")
    parser.add_argument("--fallback", type=Path, default=None,
                        help="font supplying symbols cmunrm.otf lacks (latinmodern-math.otf)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    src = args.font or find_default_font()
    if src is None or not src.is_file():
        parser.error("cmunrm.otf not found; pass --font (nixpkgs: texlivePackages.cm-unicode)")
    fallback = args.fallback or find_default_fallback()
    if fallback is None:
        print("warning: latinmodern-math.otf not found; math symbols will render as '?'", file=sys.stderr)
    text = generate(src, fallback)
    args.out.write_text(text)
    print(f"wrote {args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
