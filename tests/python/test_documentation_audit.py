"""Exercise coverage against real Doxygen output, including internals and stale inventories."""

import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

from scripts import check_documentation as audit

## Prefer an explicitly selected Doxygen, otherwise use the developer shell's executable.
DOXYGEN = os.environ.get("DOXYGEN_EXECUTABLE") or shutil.which("doxygen")
## Small maintained API with intentional omissions and a documented overload.
HEADER = """/// @file api.hpp
/// @brief Fixture API for coverage checks.
/// @brief Symbols whose documentation presence is tested.
namespace fixture {
/// @brief Execution mode.
enum class Mode {
  Ready, ///< Ready for execution.
  Running
};
/// @brief A fixture with private state.
class Example {
 public:
  /// @brief Processes an integer.
  int run(int value);
  int run(double value);
 private:
  int hidden_ = 0;
  static constexpr int limit = 4;
};
/// @brief Integer count type.
using Count = int;
}
"""
## Translation unit containing a private helper that must remain in the inventory.
SOURCE = """/// @file api.cpp
/// @brief Coverage fixture implementation.
#include "api.hpp"
namespace fixture {
static int helper(int value) { return value + 1; }
int Example::run(int value) { return helper(value); }
int Example::run(double value) { return static_cast<int>(value); }
}
"""


@unittest.skipUnless(DOXYGEN, "Doxygen is required for documentation audit integration tests")
class DocumentationAuditTests(unittest.TestCase):
    """Check that Doxygen-derived coverage cannot pass by omitting source definitions."""

    def setUp(self):
        """Generate fresh XML from a small source tree using the production extraction options."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.header = self.root / "include/api.hpp"
        self.source = self.root / "src/datamunge/api.cpp"
        for path, text in ((self.header, HEADER), (self.source, SOURCE)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        config = self.root / "Doxyfile"
        config.write_text(
            f'INPUT = "{self.root / "include"}" "{self.root / "src/datamunge"}"\n'
            f'OUTPUT_DIRECTORY = "{self.root / "output"}"\n'
            "RECURSIVE = YES\nFILE_PATTERNS = *.hpp *.cpp\n"
            "EXTRACT_ALL = YES\nEXTRACT_PRIVATE = YES\nEXTRACT_STATIC = YES\n"
            "EXTRACT_ANON_NSPACES = YES\nGENERATE_XML = YES\n"
            "INTERNAL_DOCS = YES\n"
            "GENERATE_HTML = NO\nGENERATE_LATEX = NO\nQUIET = YES\n"
        )
        subprocess.run([DOXYGEN, str(config)], cwd=self.root, check=True, capture_output=True, text=True)
        self.xml = self.root / "output/xml"

    def test_internals_enum_values_overloads_and_deduplication(self):
        """Private/static state and every enum value/overload must have independent coverage."""
        symbols = audit.read_inventory(self.xml, self.root)
        self.assertEqual(len(symbols), len({symbol.id for symbol in symbols}))
        missing = {symbol.name for symbol in symbols if not symbol.documented}
        self.assertIn("fixture::Mode::Running", missing)
        self.assertIn("fixture::Example::hidden_", missing)
        self.assertIn("fixture::Example::limit", missing)
        self.assertTrue(any("helper(" in name for name in missing))
        overloads = [symbol for symbol in symbols if symbol.name.startswith("fixture::Example::run(")]
        self.assertEqual(2, len(overloads))
        self.assertEqual(1, sum(symbol.documented for symbol in overloads))
        self.assertTrue(next(symbol for symbol in symbols if symbol.name == "fixture::Count").documented)
        self.assertTrue(next(symbol for symbol in symbols if symbol.name == "fixture::Mode::Ready").documented)

    def test_missing_compound_fails_instead_of_reducing_coverage(self):
        """A partial Doxygen run must fail even if its index file was produced."""
        next(path for path in self.xml.glob("*.xml") if path.name != "index.xml").unlink()
        with self.assertRaises(FileNotFoundError):
            audit.read_inventory(self.xml, self.root)

    def test_new_source_absent_from_xml_fails(self):
        """A new file cannot be excluded silently from the definition count."""
        (self.root / "include/new.hpp").write_text("void new_function();\n")
        with self.assertRaisesRegex(ValueError, "absent from XML"):
            audit.read_inventory(self.xml, self.root)

    def test_removed_source_still_in_xml_fails(self):
        """Deleted files in a stale output directory must not inflate documented counts."""
        self.header.unlink()
        with self.assertRaisesRegex(ValueError, "Removed files"):
            audit.read_inventory(self.xml, self.root)

    def test_source_newer_than_inventory_fails(self):
        """An old XML run cannot certify a recently edited source file."""
        later = (self.xml / "index.xml").stat().st_mtime_ns + 1_000_000_000
        os.utime(self.header, ns=(later, later))
        with self.assertRaisesRegex(ValueError, "Sources newer than XML"):
            audit.read_inventory(self.xml, self.root)

    def test_empty_index_fails(self):
        """Zero parsed definitions must not be reported as complete documentation."""
        (self.xml / "index.xml").write_text("<doxygenindex/>")
        with self.assertRaisesRegex(ValueError, "Empty source or symbol inventory"):
            audit.read_inventory(self.xml, self.root)

    def test_disabled_private_extraction_fails(self):
        """Keeping every file but hiding private members must still invalidate coverage."""
        config = self.xml / "Doxyfile.xml"
        text = config.read_text()
        tree = ET.fromstring(text)
        tree.find("./option[@id='EXTRACT_PRIVATE']/value").text = "NO"
        config.write_text(ET.tostring(tree, encoding="unicode"))
        with self.assertRaisesRegex(ValueError, "EXTRACT_PRIVATE must be YES"):
            audit.read_inventory(self.xml, self.root)

    def test_check_failure_and_unknown_path_exit_codes(self):
        """Missing descriptions and unmatched scopes must both fail the command-line check."""
        symbols = audit.read_inventory(self.xml, self.root)
        with mock.patch.object(audit, "read_inventory", return_value=symbols):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(1, audit.main(["--check", "--limit", "0"]))
                self.assertEqual(2, audit.main(["--check", "--path", "include/typo.hpp"]))
                self.assertEqual(0, audit.main(["--limit", "0"]))

    def test_path_scope_does_not_match_sibling_prefix(self):
        """A selected file or directory must not accidentally include similarly named siblings."""
        symbol = audit.Symbol("id", "file", "api.hpp", "include/api.hpp", 1, True)
        self.assertTrue(audit.in_scope(symbol, "include"))
        self.assertTrue(audit.in_scope(symbol, "include/api.hpp"))
        self.assertFalse(audit.in_scope(symbol, "include/api"))

    def test_empty_completed_manifest_cannot_pass(self):
        """Removing finished scopes must not turn the CI gate into an empty successful check."""
        manifest = self.root / "completed.txt"
        manifest.write_text("# No verified files\n")
        with mock.patch.object(audit, "COMPLETED_PATHS", manifest):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(2, audit.main(["--check", "--completed"]))


if __name__ == "__main__":
    unittest.main()
