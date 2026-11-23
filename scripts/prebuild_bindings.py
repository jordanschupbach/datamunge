#!/usr/bin/env python3
"""Generate binding sources before packaging; failed generation leaves old files intact.

All come from Rosetta: Python from src/pydatamunge/rosetta, R from src/datamunger/rosetta,
Lua from src/datamungelua/rosetta, Common Lisp from src/datamungelisp/rosetta, Julia from
src/datamungejulia/rosetta, C# from src/datamungecsharp/rosetta, COBOL from
src/datamungecobol/rosetta, JavaScript from src/datamungejs/rosetta, Fortran from
src/datamungefortran/rosetta, Ruby from src/datamungeruby/rosetta, Octave from
src/datamungeoctave/rosetta, Tcl from src/datamungetcl/rosetta, Guile from
src/datamungeguile/rosetta, Java from src/datamungejava/rosetta, Go from src/datamungego/rosetta,
OCaml from src/datamungeocaml/rosetta, GNU APL from src/datamungeapl/rosetta, Visual Basic from
src/datamungevb/rosetta, Rust from src/datamungerust/rosetta, PHP from
src/datamungephp/rosetta, Perl from src/datamungeperl/rosetta,
Swift from src/datamungeswift/rosetta, Free Pascal from src/datamungepascal/rosetta,
Dart from src/datamungedart/rosetta, Ada from src/datamungeada/rosetta,
Objective-C from src/datamungeobjc/rosetta, SWI-Prolog from src/datamungeprolog/rosetta,
Haskell from src/datamungehaskell/rosetta."""

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
FORTRAN_PARTS = 16
RUBY_PARTS = 16
OCTAVE_PARTS = 16
TCL_PARTS = 16
GUILE_PARTS = 16
JAVA_PARTS = 16
GO_PARTS = 16
OCAML_PARTS = 16
APL_PARTS = 16
VB_PARTS = 16
RUST_PARTS = 16
PHP_PARTS = 16
PERL_PARTS = 16
SWIFT_PARTS = 16
PASCAL_PARTS = 16
DART_PARTS = 16
ADA_PARTS = 16
OBJC_PARTS = 16
PROLOG_PARTS = 16
HASKELL_PARTS = 16


def sync_directory(generated, destination, pattern):
    """Make the files matching pattern under destination those under generated: files no longer
    generated are removed (with directories left empty), changed ones replaced."""
    generated_files = {f.relative_to(generated) for f in generated.rglob(pattern) if f.is_file()}
    if destination.exists():
        for old in sorted(destination.rglob("*"), reverse=True):
            if old.is_file() and old.match(pattern) and old.relative_to(destination) not in generated_files:
                old.unlink()
            elif old.is_dir() and not any(old.iterdir()):
                old.rmdir()
    for relative in sorted(generated_files):
        (destination / relative).parent.mkdir(parents=True, exist_ok=True)
        install_if_changed(generated / relative, destination / relative)


def generate_rosetta(header, policy, module, stem, parts, include_dirs=("include",), r_interface=None, lua_meta=None,
                     lisp_dir=None, julia_dir=None, csharp_dir=None, cobol_dir=None, js_types=None,
                     fortran_dir=None, ruby_lib=None, octave_dir=None, tcl_dir=None, guile_dir=None, java_dir=None,
                     go_dir=None, ocaml_dir=None, apl_dir=None, generic=None):
    """Generate src/<stem>*.cpp with Rosetta: the Python extension `module`, with
    r_interface (a path) the R package `module` and its R6 classes at that path, or with
    lua_meta (a path) the Lua module `module` and its LuaCATS stub at that path, or with
    lisp_dir (a directory) the Common Lisp system `module`: lisp_dir/module.lisp and .asd, or with
    julia_dir (a directory) the Julia package Datamunge: julia_dir/src/Datamunge.jl and Project.toml, or with
    csharp_dir (a directory) the C# assembly Datamunge: csharp_dir/Datamunge.cs and Datamunge.csproj, or with
    cobol_dir (a directory) the COBOL copybooks cobol_dir/module.cpy and module-callbacks.cpy, or with
    js_types (a path) the Node.js addon `module` and its TypeScript declarations at that path, or with
    fortran_dir (a directory) the Fortran module `module`: fortran_dir/module.f90, or with
    ruby_lib (a directory) the Ruby extension `module` and its loader ruby_lib/module.rb, or with
    octave_dir (a directory) the Octave oct-file __module__ and its package octave_dir/+module, or with
    tcl_dir (a directory) the Tcl extension `module` and its package files tcl_dir/module.tcl and pkgIndex.tcl, or with
    guile_dir (a directory) the Guile extension libmodule_guile and its modules guile_dir/module.scm and
    guile_dir/module/*.scm, or with
    java_dir (a directory) the JNI library libmodule_java and the Java sources of package `module` under
    java_dir (java_dir/module/*.java, java_dir/module/<submodule>/*.java), or with
    go_dir (a directory) the shared library libmodule_go and the Go module `module` in go_dir (go.mod,
    module.go, internal/rosetta/*.go and one subpackage per policy module, go_dir/<submodule>/*.go), or with
    ocaml_dir (a directory) the C ABI library with its OCaml C stubs and the findlib package `module` in
    ocaml_dir (module.ml, module.mli with the odoc documentation, META), or with
    apl_dir (a directory) the GNU APL native function library libmodule_gnu_apl and the GNU APL script
    apl_dir/module.apl (dm∆ functions whose ⍝ comments carry the documentation), or with
    generic a pair (arguments, outputs) for the languages added since: arguments(tmp) gives the
    language options (writing under tmp), and each (source, destination, pattern) of outputs
    synchronises tmp/source with destination as sync_directory does."""
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
        if fortran_dir is not None:
            # The native library is datamunge_fortran (libdatamunge_fortran), not libdatamunge.
            language = ["--language", "fortran", "--fortran-api", "full",
                        "--fortran-output", str(Path(tmp) / (module + ".f90"))]
        if ruby_lib is not None:
            # ruby_lib/module.rb require_relative's the native extension module/module.<DLEXT>
            # beside it and carries the YARD documentation (stubs under `if nil`).
            language = ["--language", "ruby", "--ruby-api", "full", "--ruby-module", "Datamunge",
                        "--ruby-output", str(Path(tmp) / (module + ".rb")),
                        "--ruby-native", module + "/" + module]
        if octave_dir is not None:
            # The oct-file's one DEFMETHOD_DLD is __module__ (build it as __module__.oct); the
            # +module package of classdef .m files (with the documentation) goes beside it.
            language = ["--language", "octave", "--octave-api", "full",
                        "--octave-output", str(Path(tmp) / "octave")]
        if tcl_dir is not None:
            # tcl_dir is the package directory: module.tcl loads the extension module[info
            # sharedlibextension] beside it (Module_Init) and holds the documentation stubs.
            language = ["--language", "tcl", "--tcl-api", "full",
                        "--tcl-output", str(Path(tmp) / (module + ".tcl")),
                        "--tcl-pkgindex", str(Path(tmp) / "pkgIndex.tcl"),
                        "--tcl-version", "0.0.1"]
        if guile_dir is not None:
            # guile_dir/module.scm (with the policy submodules guile_dir/module/*.scm) is the module
            # (module); it load-extension's libmodule_guile from its own directory and holds the
            # documented GOOPS API (docstrings).
            language = ["--language", "guile", "--guile-api", "full", "--guile-module", module,
                        "--guile-extension", "lib" + module + "_guile",
                        "--guile-output", str(Path(tmp) / "guile" / (module + ".scm"))]
        if java_dir is not None:
            # One documented .java file per class under java_dir/module/ (Javadoc); the classes load
            # the JNI library libmodule_java (libmodule is the C++ library itself).
            language = ["--language", "java", "--java-api", "full", "--java-package", module,
                        "--java-library", module + "_java",
                        "--java-output", str(Path(tmp) / "java")]
        if go_dir is not None:
            # The Go module `module` (go.mod, module.go, internal/rosetta, one subpackage per policy
            # module) with godoc comments; cgo links the shared library libmodule_go built from the parts
            # (libmodule is the C++ library itself).
            language = ["--language", "go", "--go-api", "full", "--go-package", module,
                        "--go-module-path", module, "--go-library", module + "_go",
                        "--go-output", str(Path(tmp) / "go" / (module + ".go"))]
        if ocaml_dir is not None:
            # ocaml_dir is the findlib package directory: module.ml (externals into the stubs
            # dllmodule_stubs/libmodule_stubs built from the parts), module.mli (odoc) and META.
            language = ["--language", "ocaml", "--ocaml-api", "full",
                        "--ocaml-output", str(Path(tmp) / "ocaml" / (module + ".ml"))]
        if apl_dir is not None:
            # One GNU APL native function library (libmodule_gnu_apl.so) and the script module.apl that
            # ⎕FX's it: from the current directory, or the file named by DATAMUNGE_APL_LIBRARY (set
            # by the dev shells and the justfile), so the committed script holds no build paths.
            language = ["--language", "apl", "--apl-api", "full", "--apl-prefix", "dm", "--apl-library-dir", ".",
                        "--gnu-apl-output", str(Path(tmp) / "apl" / (module + ".apl"))]
        if generic is not None:
            language = generic[0](Path(tmp))
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
        if fortran_dir is not None:
            (ROOT / fortran_dir).mkdir(parents=True, exist_ok=True)
            install_if_changed(Path(tmp) / (module + ".f90"), ROOT / fortran_dir / (module + ".f90"))
        if ruby_lib is not None:
            (ROOT / ruby_lib).mkdir(parents=True, exist_ok=True)
            install_if_changed(Path(tmp) / (module + ".rb"), ROOT / ruby_lib / (module + ".rb"))
        if octave_dir is not None:
            # Rosetta does not delete .m files of an earlier generation, so the package directory is
            # synchronised: files no longer generated go, changed ones are replaced.
            sync_directory(Path(tmp) / "octave" / ("+" + module), ROOT / octave_dir / ("+" + module), "*")
        if tcl_dir is not None:
            (ROOT / tcl_dir).mkdir(parents=True, exist_ok=True)
            for name in (module + ".tcl", "pkgIndex.tcl"):
                install_if_changed(Path(tmp) / name, ROOT / tcl_dir / name)
        if guile_dir is not None:
            # As for Octave, guile_dir is synchronised: .scm files no longer generated go.
            sync_directory(Path(tmp) / "guile", ROOT / guile_dir, "*.scm")
        if java_dir is not None:
            # As for Octave, java_dir is synchronised: .java files no longer generated go.
            sync_directory(Path(tmp) / "java", ROOT / java_dir, "*.java")
        if go_dir is not None:
            # As for Octave, go_dir is synchronised: files no longer generated go.
            sync_directory(Path(tmp) / "go", ROOT / go_dir, "*")
        if ocaml_dir is not None:
            # As for Octave, ocaml_dir is synchronised: files no longer generated go.
            sync_directory(Path(tmp) / "ocaml", ROOT / ocaml_dir, "*")
        if apl_dir is not None:
            (ROOT / apl_dir).mkdir(parents=True, exist_ok=True)
            install_if_changed(Path(tmp) / "apl" / (module + ".apl"), ROOT / apl_dir / (module + ".apl"))
        if generic is not None:
            for source, destination, pattern in generic[1]:
                sync_directory(Path(tmp) / source, ROOT / destination, pattern)


def prebuild_python():
    """Generate src/datamunge_python_wrap*.cpp, the pydatamunge._datamunge extension."""
    bindings = Path("src/pydatamunge/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "pydatamunge._datamunge",
                     "datamunge_python_wrap", PYTHON_PARTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["python", "r", "lua", "lisp", "julia", "csharp", "cobol", "javascript", "fortran", "ruby", "octave", "tcl", "guile", "java", "go", "ocaml", "apl", "vb", "rust", "php", "perl", "swift", "pascal", "dart", "ada", "objc", "prolog", "haskell"])
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
    if args.language == "fortran":
        prebuild_fortran()
        return
    if args.language == "ruby":
        prebuild_ruby()
        return
    if args.language == "octave":
        prebuild_octave()
        return
    if args.language == "tcl":
        prebuild_tcl()
        return
    if args.language == "guile":
        prebuild_guile()
        return
    if args.language == "java":
        prebuild_java()
        return
    if args.language == "go":
        prebuild_go()
        return
    if args.language == "ocaml":
        prebuild_ocaml()
        return
    if args.language == "apl":
        prebuild_apl()
    if args.language == "vb":
        prebuild_vb()
    if args.language == "rust":
        prebuild_rust()
    if args.language == "php":
        prebuild_php()
    if args.language == "perl":
        prebuild_perl()
    if args.language == "swift":
        prebuild_swift()
    if args.language == "pascal":
        prebuild_pascal()
    if args.language == "dart":
        prebuild_dart()
    if args.language == "ada":
        prebuild_ada()
    if args.language == "objc":
        prebuild_objc()
    if args.language == "prolog":
        prebuild_prolog()
    if args.language == "haskell":
        prebuild_haskell()
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


def prebuild_fortran():
    """Generate src/datamunge_fortran_wrap*.cpp (the native library libdatamunge_fortran) and the
    Fortran 2018 module src/datamungefortran/datamunge.f90 (modules datamunge, datamunge_rosetta and
    datamunge_data): derived types and type-bound procedures whose FORD `!>` comments carry the C++
    documentation (mep's Fortran LSP reads them from the committed source)."""
    bindings = Path("src/datamungefortran/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_fortran_wrap", FORTRAN_PARTS,
                     fortran_dir="src/datamungefortran")


def prebuild_ruby():
    """Generate src/datamunge_ruby_wrap*.cpp (the MRI extension datamunge/datamunge.<DLEXT>) and
    src/datamungeruby/lib/datamunge.rb, which loads it (`require 'datamunge'`) and holds YARD
    documentation for module Datamunge (yard, mep's Ruby LSP)."""
    bindings = Path("src/datamungeruby/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_ruby_wrap", RUBY_PARTS,
                     ruby_lib="src/datamungeruby/lib")


def prebuild_octave():
    """Generate src/datamunge_octave_wrap*.cpp (the oct-file __datamunge__.oct) and the Octave
    package src/datamungeoctave/inst/+datamunge: one classdef or function .m file per class, enum,
    function and constant, whose help text carries the C++ documentation (`help
    datamunge.DataFrame.filter`, mep's Octave LSP through MEP_OCTAVE_PATH)."""
    bindings = Path("src/datamungeoctave/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_octave_wrap", OCTAVE_PARTS,
                     octave_dir="src/datamungeoctave/inst")


def prebuild_tcl():
    """Generate src/datamunge_tcl_wrap*.cpp (the Tcl stubs extension datamunge[info
    sharedlibextension], Datamunge_Init) and the package files src/datamungetcl/pkg/datamunge/
    datamunge.tcl and pkgIndex.tcl (`package require datamunge`): TclOO classes in ::datamunge, with
    the documentation as commented proc/oo::class stubs (mep's Tcl LSP through MEP_TCL_PATH)."""
    bindings = Path("src/datamungetcl/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_tcl_wrap", TCL_PARTS,
                     tcl_dir="src/datamungetcl/pkg/datamunge")


def prebuild_guile():
    """Generate src/datamunge_guile_wrap*.cpp (the libguile extension libdatamunge_guile) and the
    module src/datamungeguile/scheme/datamunge.scm (plus the policy submodule datamunge/data.scm):
    GOOPS classes and generics whose docstrings carry the C++ documentation (`(use-modules
    (datamunge))`, procedure-documentation, mep's Scheme LSP through MEP_GUILE_PATH)."""
    bindings = Path("src/datamungeguile/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_guile_wrap", GUILE_PARTS,
                     guile_dir="src/datamungeguile/scheme")


def prebuild_java():
    """Generate src/datamunge_java_wrap*.cpp (the JNI library libdatamunge_java) and the Java sources
    of package datamunge under src/datamungejava/java (datamunge/*.java, datamunge/data/Data.java):
    AutoCloseable classes whose Javadoc carries the C++ documentation (javadoc, mep's Java server
    through MEP_JAVA_SOURCEPATH)."""
    bindings = Path("src/datamungejava/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_java_wrap", JAVA_PARTS,
                     java_dir="src/datamungejava/java")


def prebuild_go():
    """Generate src/datamunge_go_wrap*.cpp (the shared library libdatamunge_go that cgo links) and the Go
    module datamunge under src/datamungego/go/datamunge (go.mod, datamunge.go, the policy subpackage
    data/ and internal/rosetta): structs with Close and godoc comments carrying the C++
    documentation (go doc, mep's Go server through a go.mod replace)."""
    bindings = Path("src/datamungego/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_go_wrap", GO_PARTS,
                     go_dir="src/datamungego/go/datamunge")


def prebuild_ocaml():
    """Generate src/datamunge_ocaml_wrap*.cpp (the C ABI library with its OCaml C stubs, built as
    dlldatamunge_stubs and libdatamunge_stubs) and the findlib package src/datamungeocaml/ocaml/datamunge
    (datamunge.ml, META and datamunge.mli, whose odoc comments carry the C++ documentation: odoc, mep's
    OCaml server through MEP_OCAML_PATH). Classes are submodules (Datamunge.DataFrame)."""
    bindings = Path("src/datamungeocaml/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_ocaml_wrap", OCAML_PARTS,
                     ocaml_dir="src/datamungeocaml/ocaml/datamunge")



def prebuild_apl():
    """Generate src/datamunge_apl_wrap*.cpp (the GNU APL native function library libdatamunge_gnu_apl)
    and the GNU APL script src/datamungeapl/apl/datamunge.apl, which ⎕FX's that library and defines one
    dm∆ function per C++ function, constructor, method and field; their ⍝ comments carry the C++
    documentation (⎕CR, mep's APL server through MEP_APL_PATH)."""
    bindings = Path("src/datamungeapl/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_apl_wrap", APL_PARTS,
                     apl_dir="src/datamungeapl/apl")


def prebuild_vb():
    """Generate src/datamunge_vb_wrap*.cpp (the native library datamunge_vb) and the Visual Basic
    project src/datamungevb (Datamunge.vb, Datamunge.vbproj), whose XML doc comments carry the C++
    documentation (IntelliSense, mep's VB server through MEP_VB_PATH)."""
    bindings = Path("src/datamungevb/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_vb_wrap", VB_PARTS,
                     generic=(lambda tmp: ["--language", "visual-basic", "--vb-namespace", "Datamunge",
                                           "--vb-library", "datamunge_vb",
                                           "--vb-output", str(tmp / "vb" / "Datamunge.vb"),
                                           "--vb-project", str(tmp / "vb" / "Datamunge.vbproj")],
                              [("vb", "src/datamungevb", "Datamunge.vb"), ("vb", "src/datamungevb", "Datamunge.vbproj")]))


def prebuild_rust():
    """Generate src/datamunge_rust_wrap*.cpp (the shared library libdatamunge_rust) and the Cargo crate
    src/datamungerust/datamunge (Cargo.toml, build.rs, src/lib.rs with the rustdoc documentation,
    src/rosetta.rs); the policy module datamunge::data is datamunge::data."""
    bindings = Path("src/datamungerust/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_rust_wrap", RUST_PARTS,
                     generic=(lambda tmp: ["--language", "rust", "--rust-crate", "datamunge",
                                           "--rust-library", "datamunge_rust",
                                           "--rust-output", str(tmp / "rust" / "datamunge")],
                              [("rust/datamunge", "src/datamungerust/datamunge", "*")]))


def prebuild_php():
    """Generate src/datamunge_php_wrap*.cpp (the shared library libdatamunge_php) and the PHP file
    src/datamungephp/php/datamunge.php (namespace Datamunge over PHP's FFI extension), whose PHPDoc
    comments carry the C++ documentation."""
    bindings = Path("src/datamungephp/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_php_wrap", PHP_PARTS,
                     generic=(lambda tmp: ["--language", "php", "--php-namespace", "Datamunge",
                                           "--php-library", "datamunge_php",
                                           "--php-output", str(tmp / "php" / "datamunge.php")],
                              [("php", "src/datamungephp/php", "*.php")]))


def prebuild_perl():
    """Generate src/datamunge_perl_wrap*.cpp (the shared library libdatamunge_perl, whose first part
    carries the XS glue) and the module src/datamungeperl/lib/Datamunge.pm, documented in POD
    (perldoc Datamunge, mep's Perl server through MEP_PERL_PATH)."""
    bindings = Path("src/datamungeperl/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_perl_wrap", PERL_PARTS,
                     generic=(lambda tmp: ["--language", "perl", "--perl-package", "Datamunge",
                                           "--perl-library", "datamunge_perl",
                                           "--perl-output", str(tmp / "perl" / "Datamunge.pm")],
                              [("perl", "src/datamungeperl/lib", "*.pm")]))


def prebuild_swift():
    """Generate src/datamunge_swift_wrap*.cpp (the shared library libdatamunge_swift) and the Swift
    package src/datamungeswift/Datamunge (Package.swift, Sources/Datamunge/{Datamunge,Rosetta}.swift
    with the DocC documentation, and the C module Sources/CDatamunge)."""
    bindings = Path("src/datamungeswift/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_swift_wrap", SWIFT_PARTS,
                     generic=(lambda tmp: ["--language", "swift", "--swift-module", "Datamunge",
                                           "--swift-library", "datamunge_swift",
                                           "--swift-output", str(tmp / "swift" / "Datamunge")],
                              [("swift/Datamunge", "src/datamungeswift/Datamunge", "*")]))


def prebuild_pascal():
    """Generate src/datamunge_pascal_wrap*.cpp (the shared library libdatamunge_pascal) and the Free
    Pascal unit src/datamungepascal/datamunge.pas, whose XML doc comments carry the C++ documentation."""
    bindings = Path("src/datamungepascal/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_pascal_wrap", PASCAL_PARTS,
                     generic=(lambda tmp: ["--language", "pascal", "--pascal-unit", "datamunge",
                                           "--pascal-library", "datamunge_pascal",
                                           "--pascal-output", str(tmp / "pascal" / "datamunge.pas")],
                              [("pascal", "src/datamungepascal/pascal", "*.pas")]))


def prebuild_dart():
    """Generate src/datamunge_dart_wrap*.cpp (the shared library libdatamunge_dart) and the Dart package
    src/datamungedart/datamunge (pubspec.yaml, lib/datamunge.dart with the dartdoc documentation,
    lib/src/rosetta.dart), loaded through dart:ffi."""
    bindings = Path("src/datamungedart/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_dart_wrap", DART_PARTS,
                     generic=(lambda tmp: ["--language", "dart", "--dart-package", "datamunge",
                                           "--dart-library", "datamunge_dart",
                                           "--dart-output", str(tmp / "dart" / "datamunge")],
                              [("dart/datamunge", "src/datamungedart/datamunge", "*")]))


def prebuild_ada():
    """Generate src/datamunge_ada_wrap*.cpp (the shared library libdatamunge_ada) and the Ada package
    src/datamungeada/ada (datamunge.ads with the GNATdoc documentation, datamunge.adb, datamunge.gpr)."""
    bindings = Path("src/datamungeada/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_ada_wrap", ADA_PARTS,
                     generic=(lambda tmp: ["--language", "ada", "--ada-package", "Datamunge",
                                           "--ada-library", "datamunge_ada", "--ada-output", str(tmp / "ada")],
                              [("ada", "src/datamungeada/ada", "datamunge.*")]))


def prebuild_objc():
    """Generate src/datamunge_objc_wrap*.cpp (the shared library libdatamunge_objc) and the Objective-C
    classes src/datamungeobjc/objc (DM.h with the /// documentation, DM.m with the runtime)."""
    bindings = Path("src/datamungeobjc/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_objc_wrap", OBJC_PARTS,
                     generic=(lambda tmp: ["--language", "objective-c", "--objc-prefix", "DM",
                                           "--objc-library", "datamunge_objc", "--objc-output", str(tmp / "objc")],
                              [("objc", "src/datamungeobjc/objc", "DM.*")]))


def prebuild_prolog():
    """Generate src/datamunge_prolog_wrap*.cpp (the foreign library libdatamunge_prolog) and the SWI-Prolog
    module src/datamungeprolog/prolog/datamunge.pl (predicates, PlDoc and the runtime)."""
    bindings = Path("src/datamungeprolog/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_prolog_wrap", PROLOG_PARTS,
                     generic=(lambda tmp: ["--language", "prolog", "--prolog-module", "datamunge",
                                           "--prolog-library", "datamunge_prolog",
                                           "--prolog-output", str(tmp / "prolog" / "datamunge.pl")],
                              [("prolog", "src/datamungeprolog/prolog", "datamunge.pl")]))


def prebuild_haskell():
    """Generate src/datamunge_haskell_wrap*.cpp (the shared library libdatamunge_haskell) and the GHC modules
    src/datamungehaskell/haskell/Datamunge.hs (Haddock documentation) and Datamunge/Rosetta.hs (the FFI runtime)."""
    bindings = Path("src/datamungehaskell/rosetta")
    generate_rosetta(bindings / "bindings.hpp", bindings / "policy.json", "datamunge", "datamunge_haskell_wrap", HASKELL_PARTS,
                     generic=(lambda tmp: ["--language", "haskell", "--haskell-module", "Datamunge",
                                           "--haskell-output", str(tmp / "haskell" / "Datamunge.hs")],
                              [("haskell", "src/datamungehaskell/haskell", "Datamunge.hs"),
                               ("haskell/Datamunge", "src/datamungehaskell/haskell/Datamunge", "*.hs")]))


if __name__ == "__main__":
    main()
