#!/usr/bin/env python3
"""Generate binding sources before packaging; failed generation leaves old files intact.

All come from Rosetta: Python from src/pydatamunge/rosetta, R from src/datamunger/rosetta,
Lua from src/datamungelua/rosetta, Common Lisp from src/datamungelisp/rosetta."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install_if_changed(source, destination):
    if not destination.exists() or source.read_bytes() != destination.read_bytes():
        source.replace(destination)


# The generated extension is split into this many sources so they compile in parallel
# (one source took over a minute; each part mostly re-parses the shared headers).
PYTHON_PARTS = 16
R_PARTS = 16
LUA_PARTS = 16
LISP_PARTS = 16


def generate_rosetta(header, policy, module, stem, parts, include_dirs=("include",), r_interface=None, lua_meta=None,
                     lisp_dir=None):
    """Generate src/<stem>*.cpp with Rosetta: the Python extension `module`, with
    r_interface (a path) the R package `module` and its R6 classes at that path, or with
    lua_meta (a path) the Lua module `module` and its LuaCATS stub at that path, or with
    lisp_dir (a directory) the Common Lisp system `module`: lisp_dir/module.lisp and .asd."""
    with tempfile.TemporaryDirectory(prefix=".rosetta-", dir=ROOT / "src") as tmp:
        generated = Path(tmp) / (stem + ".cpp")
        language = ["--language", "python"]
        if r_interface is not None:
            language = ["--language", "r", "--r-package", "--r-output", str(Path(tmp) / Path(r_interface).name)]
        if lua_meta is not None:
            language = ["--language", "lua", "--lua-api", "full", "--lua-meta", str(Path(tmp) / Path(lua_meta).name)]
        if lisp_dir is not None:
            language = ["--language", "common-lisp", "--lisp-api", "full", "--lisp-nicknames", "dm",
                        "--lisp-output", str(Path(tmp) / (module + ".lisp")),
                        "--lisp-system", str(Path(tmp) / (module + ".asd"))]
        subprocess.run(
            [
                os.environ.get("ROSETTA", "rosetta"),
                str(header),
                *language,
                "--module", module,
                "--policy", str(policy),
                # Relative to src/<stem>.cpp, so sdists build anywhere.
                "--include-as", Path(header).relative_to("src").as_posix(),
                "--skip-function-bodies",
                "--skip-unsupported",
                "--parts", str(parts),
                "-o", str(generated),
                "--", "-std=c++23", *("-I" + d for d in include_dirs),
            ],
            cwd=ROOT,
            check=True,
        )
        generated_parts = sorted(Path(tmp).glob(stem + "*.cpp"))
        for part in generated_parts:
            install_if_changed(part, ROOT / "src" / part.name)
        # Drop parts left over from a run with more of them.
        current = {part.name for part in generated_parts}
        for stale in (ROOT / "src").glob(stem + "*.cpp"):
            if stale.name not in current:
                stale.unlink()
        # The former SWIG directors header.
        (ROOT / "src" / (stem + ".h")).unlink(missing_ok=True)
        if r_interface is not None:
            install_if_changed(Path(tmp) / Path(r_interface).name, ROOT / r_interface)
        if lua_meta is not None:
            (ROOT / lua_meta).parent.mkdir(parents=True, exist_ok=True)
            install_if_changed(Path(tmp) / Path(lua_meta).name, ROOT / lua_meta)
        if lisp_dir is not None:
            for name in (module + ".lisp", module + ".asd"):
                install_if_changed(Path(tmp) / name, ROOT / lisp_dir / name)


def prebuild_python():
    """Generate src/datamunge_python_wrap*.cpp, the pydatamunge._datamunge extension."""
    bindings = Path("src/pydatamunge/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "pydatamunge._datamunge",
                     "datamunge_python_wrap", PYTHON_PARTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["python", "r", "lua", "lisp"])
    args = parser.parse_args()
    if args.language == "python":
        prebuild_python()
        return
    if args.language == "lua":
        prebuild_lua()
        return
    if args.language == "lisp":
        prebuild_lisp()
        return
    prebuild_r()


def roxygenize(package):
    """Write man/ pages from the roxygen comments Rosetta generates from the C++ Doxygen comments.

    Only the Rd roclet runs (NAMESPACE stays hand-written) and the R files are sourced, not the
    compiled package loaded. Without roxygen2 the package simply has no generated help pages."""
    script = (
        'if (!requireNamespace("roxygen2", quietly = TRUE)) {'
        ' message("roxygen2 is not installed: skipping generated help pages"); quit(status = 0L) };'
        ' roxygen2::roxygenise(commandArgs(TRUE)[[1L]], roclets = "rd", load_code = roxygen2::load_source)'
    )
    subprocess.run(["Rscript", "--vanilla", "-e", script, str(package)], cwd=ROOT, check=True)


def prebuild_r():
    """Generate src/datamunge_r_wrap*.cpp and R/datamunger.R, the datamunger package's R6 classes, and their help pages."""
    bindings = Path("src/datamunger/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunger", "datamunge_r_wrap", R_PARTS,
                     r_interface="R/datamunger.R")
    roxygenize(ROOT)


def prebuild_lua():
    """Generate src/datamunge_lua_wrap*.cpp, the `datamunge` Lua module, and its LuaCATS stub
    src/datamungelua/library/datamunge.lua (editor hover and completion; see .luarc.json)."""
    bindings = Path("src/datamungelua/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_lua_wrap", LUA_PARTS,
                     lua_meta="src/datamungelua/library/datamunge.lua")



def prebuild_lisp():
    """Generate src/datamunge_lisp_wrap*.cpp (the native library datamunge.so) and the
    `datamunge` Common Lisp system, src/datamungelisp/datamunge.{lisp,asd}: CLOS classes and
    generic functions whose docstrings carry the C++ documentation (package nickname `dm`)."""
    bindings = Path("src/datamungelisp/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_lisp_wrap", LISP_PARTS,
                     lisp_dir="src/datamungelisp")


if __name__ == "__main__":
    main()
