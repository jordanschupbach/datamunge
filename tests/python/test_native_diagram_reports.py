"""Keep executable documentation figures on the canonical native graphics path."""
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "docs/mepml/examples"
BLOCK = re.compile(r'^```\{([^\n]+)\}\n(.*?)^```', re.M | re.S)


class NativeDiagramReports(unittest.TestCase):
    def test_no_external_diagram_renderer_or_source(self):
        for report in REPORTS.rglob("*.mepml"):
            with self.subTest(report=report):
                source = report.read_text()
                self.assertNotRegex(source, r"(?i)tikz|dvisvgm")
        for extension in ("*.tikz", "*.tex"):
            sources = [p for p in REPORTS.rglob(extension) if "build" not in p.relative_to(REPORTS).parts]
            self.assertEqual(sources, [], f"External diagram sources remain: {sources}")

    def test_native_blocks_are_reproducible_and_have_saved_results(self):
        found = 0
        for report in REPORTS.rglob("*.mepml"):
            source = report.read_text()
            for match in BLOCK.finditer(source):
                attributes, code = match.groups()
                if "dm::graphics::Diagram" not in code:
                    continue
                found += 1
                with self.subTest(report=report):
                    self.assertTrue(attributes.startswith("cpp,"))
                    self.assertTrue(code.startswith("#include <dm/dm.hpp>"))
                    self.assertIn("pkg-config --cflags --libs datamunge", attributes)
                    figure = re.search(r'file="([^"]+)"', attributes)
                    if figure is None:
                        # Multi-figure blocks emit their own markdown and save every
                        # image through a build/ directory constant.
                        self.assertIn('results="markdown"', attributes)
                        self.assertRegex(code, r'const std::string directory="build/')
                        self.assertRegex(code, r"\.save_svg\(directory\+")
                        continue
                    self.assertRegex(code, r'\.save_svg\("build/')
                    figure = figure.group(1)
                    self.assertIn(f"\\image({figure})", source[match.end():])
                    # Rendered figures are gitignored and only exist after `just docs`,
                    # so check their content when present.
                    image = report.parent / figure
                    if not image.is_file():
                        continue
                    svg = ET.parse(image).getroot()
                    self.assertEqual(svg.tag, "{http://www.w3.org/2000/svg}svg")
                    self.assertEqual(svg.attrib.get("role"), "img")
                    self.assertTrue(any("aria-label" in node.attrib for node in svg.iter()))
        self.assertGreater(found, 0)

    def test_saved_figures_have_no_external_diagram_provenance(self):
        for image in REPORTS.rglob("*.svg"):
            if "build" in image.relative_to(REPORTS).parts:
                continue
            with self.subTest(image=image):
                self.assertNotIn("dvisvgm", image.read_text()[:2000])


if __name__ == "__main__":
    unittest.main()
