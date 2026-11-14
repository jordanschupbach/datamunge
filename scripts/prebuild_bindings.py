#!/usr/bin/env python3
"""Generate binding sources before packaging; failed generation leaves old files intact.

All come from Rosetta: Python from src/pydatamunge/rosetta, R from src/datamunger/rosetta,
Lua from src/datamungelua/rosetta, Common Lisp from src/datamungelisp/rosetta, Julia from
src/datamungejulia/rosetta, C# from src/datamungecsharp/rosetta, COBOL from
src/datamungecobol/rosetta, JavaScript from src/datamungejs/rosetta."""

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
JULIA_PARTS = 16
CSHARP_PARTS = 16
COBOL_PARTS = 16
JS_PARTS = 16


def generate_rosetta(header, policy, module, stem, parts, include_dirs=("include",), r_interface=None, lua_meta=None,
                     lisp_dir=None, julia_dir=None, csharp_dir=None, cobol_dir=None, js_types=None):
    """Generate src/<stem>*.cpp with Rosetta: the Python extension `module`, with
    r_interface (a path) the R package `module` and its R6 classes at that path, or with
    lua_meta (a path) the Lua module `module` and its LuaCATS stub at that path, or with
    lisp_dir (a directory) the Common Lisp system `module`: lisp_dir/module.lisp and .asd, or with
    julia_dir (a directory) the Julia package Datamunge: julia_dir/src/Datamunge.jl and Project.toml, or with
    csharp_dir (a directory) the C# assembly Datamunge: csharp_dir/Datamunge.cs and Datamunge.csproj, or with
    cobol_dir (a directory) the COBOL copybooks cobol_dir/module.cpy and module-callbacks.cpy, or with
    js_types (a path) the Node.js addon `module` and its TypeScript declarations at that path."""
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
        if julia_dir is not None:
            language = ["--language", "julia", "--julia-api", "full", "--julia-package", "Datamunge",
                        "--julia-output", str(Path(tmp) / "Datamunge.jl"),
                        "--julia-project", str(Path(tmp) / "Project.toml")]
        if csharp_dir is not None:
            # The native library is datamunge_csharp, not libdatamunge (the C++ library itself).
            language = ["--language", "csharp", "--csharp-api", "full", "--csharp-namespace", "Datamunge",
                        "--csharp-library", "datamunge_csharp",
                        "--csharp-output", str(Path(tmp) / "Datamunge.cs"),
                        "--csharp-project", str(Path(tmp) / "Datamunge.csproj")]
        if cobol_dir is not None:
            # String results (summaries, to_string) get 16 KiB; other strings the default 256 bytes.
            language = ["--language", "cobol", "--cobol-api", "full", "--cobol-prefix", "dm", "--cobol-result-size", "16384",
                        "--cobol-output", str(Path(tmp) / (module + ".cpy")),
                        "--cobol-callbacks", str(Path(tmp) / (module + "-callbacks.cpy"))]
        if js_types is not None:
            language = ["--language", "javascript", "--js-types", str(Path(tmp) / Path(js_types).name)]
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
        if julia_dir is not None:
            (ROOT / julia_dir / "src").mkdir(parents=True, exist_ok=True)
            install_if_changed(Path(tmp) / "Datamunge.jl", ROOT / julia_dir / "src" / "Datamunge.jl")
            install_if_changed(Path(tmp) / "Project.toml", ROOT / julia_dir / "Project.toml")
        if csharp_dir is not None:
            (ROOT / csharp_dir).mkdir(parents=True, exist_ok=True)
            for name in ("Datamunge.cs", "Datamunge.csproj"):
                install_if_changed(Path(tmp) / name, ROOT / csharp_dir / name)
        if cobol_dir is not None:
            (ROOT / cobol_dir).mkdir(parents=True, exist_ok=True)
            for name in (module + ".cpy", module + "-callbacks.cpy"):
                install_if_changed(Path(tmp) / name, ROOT / cobol_dir / name)
        if js_types is not None:
            install_if_changed(Path(tmp) / Path(js_types).name, ROOT / js_types)


def prebuild_python():
    """Generate src/datamunge_python_wrap*.cpp, the pydatamunge._datamunge extension."""
    bindings = Path("src/pydatamunge/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "pydatamunge._datamunge",
                     "datamunge_python_wrap", PYTHON_PARTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["python", "r", "lua", "lisp", "julia", "csharp", "cobol", "javascript"])
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
    if args.language == "julia":
        prebuild_julia()
        return
    if args.language == "csharp":
        prebuild_csharp()
        return
    if args.language == "cobol":
        prebuild_cobol()
        return
    if args.language == "javascript":
        prebuild_javascript()
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


def prebuild_julia():
    """Generate src/datamunge_julia_wrap*.cpp (the native library datamunge.so) and the Julia
    package src/datamungejulia (Project.toml, src/Datamunge.jl): abstract types and generic
    functions whose Markdown docstrings carry the C++ documentation (`?fit` in the REPL, mep)."""
    bindings = Path("src/datamungejulia/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_julia_wrap", JULIA_PARTS,
                     julia_dir="src/datamungejulia")


def prebuild_csharp():
    """Generate src/datamunge_csharp_wrap*.cpp (the native library datamunge_csharp) and the C#
    project src/datamungecsharp (Datamunge.cs, Datamunge.csproj): classes, overloads and properties
    whose XML documentation comments carry the C++ documentation (IDEs, mep's C# LSP)."""
    bindings = Path("src/datamungecsharp/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_csharp_wrap", CSHARP_PARTS,
                     csharp_dir="src/datamungecsharp")


def prebuild_cobol():
    """Generate src/datamunge_cobol_wrap*.cpp (the native library libdatamungecobol) and the
    copybooks src/datamungecobol/copybooks/datamunge{,-callbacks}.cpy: one record per call (prefix
    DM), whose *>> comments carry the C++ documentation and the CALL statement."""
    bindings = Path("src/datamungecobol/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_cobol_wrap", COBOL_PARTS,
                     cobol_dir="src/datamungecobol/copybooks")


def prebuild_javascript():
    """Generate src/datamunge_js_wrap*.cpp (the Node-API addon datamungejs.node, built by
    setup.js and node-gyp) and index.d.ts beside index.js: TypeScript declarations whose JSDoc
    carries the C++ documentation, so editors (tsserver, mep) show it on hover."""
    bindings = Path("src/datamungejs/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamungejs", "datamunge_js_wrap", JS_PARTS,
                     js_types="index.d.ts")


if __name__ == "__main__":
    main()
