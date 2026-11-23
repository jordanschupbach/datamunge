TARGET := "dataframe_ex"
BENCH_TARGET := "linalg_bench"
JOBS := "20"
NN_JOBS := env_var_or_default("DATAMUNGE_EXAMPLE_BUILD_JOBS", "4")

# Set DATAMUNGE_SWIG_SRC to a local checkout of the swig fork (e.g. ~/projects/swig) to build
# against it instead of the github:jordanschupbach/swig revision pinned in flake.lock.
SWIG_SRC := env_var_or_default("DATAMUNGE_SWIG_SRC", "")
SWIG_OVERRIDE := if SWIG_SRC == "" { "" } else { " --override-input swig git+file://" + SWIG_SRC }
# Likewise DATAMUNGE_ROSETTA_SRC: a Rosetta checkout to generate the Python bindings with,
# instead of the github:jordanschupbach/rosetta revision in flake.lock (no relock needed after
# editing that checkout). Unset, a checkout next to this one (../rosetta) is used when there is
# one; "none" forces the pinned revision.
ROSETTA_SIBLING := clean(justfile_directory() / ".." / "rosetta")
ROSETTA_SRC_ENV := env_var_or_default("DATAMUNGE_ROSETTA_SRC", "")
ROSETTA_SRC := if ROSETTA_SRC_ENV == "none" { "" } else if ROSETTA_SRC_ENV != "" { ROSETTA_SRC_ENV } else if path_exists(ROSETTA_SIBLING / "flake.nix") == "true" { ROSETTA_SIBLING } else { "" }
ROSETTA_OVERRIDE := if ROSETTA_SRC == "" { "" } else { " --override-input rosetta git+file://" + ROSETTA_SRC }
NIX_DEVELOP := "nix develop --accept-flake-config --option eval-cache false" + SWIG_OVERRIDE + ROSETTA_OVERRIDE
# PKG_CONFIG_PATH the docs' mepml report blocks build against (empty: the shell's own;
# a directory with a datamunge.pc for the working tree makes them use the current library).
MEPML_PKG_CONFIG_PATH := env_var_or_default("DATAMUNGE_MEPML_PKG_CONFIG_PATH", "")
BINDINGS_DIR := "src"
# Compiles a free-format COBOL program against build/cobol/prefix (just build-cobol): the
# copybooks come from COBCPY (set by the dev shells) and the stubs from libdatamungecobol.
COBC := 'cobc -x -free -fstatic-call -Wall -I "$(pwd)/build/cobol/prefix/share/cobol/copy" -L "$(pwd)/build/cobol/prefix/lib" -ldatamungecobol -Q -Wl,-rpath,"$(pwd)/build/cobol/prefix/lib"'
# Compiles a Fortran program against build/fortran/prefix (just build-fortran): `use datamunge` finds
# the .mod files through -I, and the program links the module library and the native library (rpath).
# Append the source and `-o <exe>`; build/fortran/mod keeps the program's own .mod files out of the tree.
FORTRAN_FC := 'mkdir -p build/fortran/bin build/fortran/mod && gfortran -std=f2018 -O1 -Wall -Wno-compare-reals -I "$(pwd)/build/fortran/prefix/include/datamunge/fortran" -J build/fortran/mod -L "$(pwd)/build/fortran/prefix/lib" -Wl,-rpath,"$(pwd)/build/fortran/prefix/lib"'
FORTRAN_LIBS := '-ldatamunge_fortran_mod -ldatamunge_fortran'
# Runs Ruby with build/ruby/prefix/lib (just build-ruby) first on the load path: `require 'datamunge'`
# loads datamunge.rb there, which require_relative's the extension datamunge/datamunge.so.
RUBY := 'ruby -I "$(pwd)/build/ruby/prefix/lib"'
# Cargo builds against build/rust/prefix/lib (just build-rust): the crate's build.rs adds
# DATAMUNGE_RUST_LIB_DIR to the link search path, and the rpath lets the programs find libdatamunge_rust.
# Compiles a Swift program against build/swift/prefix (just build-swift): `import Datamunge` finds the
# module in lib/swift and its C module in share/swift/CDatamunge; the program links libDatamunge and
# the Swift runtime with rpaths. Append the source and `-o <exe>`.
SWIFTC := 'mkdir -p build/swift/bin && swift_root="${DATAMUNGE_SWIFT_ROOT:-$(dirname "$(dirname "$(command -v swiftc)")")}" && "$swift_root/bin/swiftc" -swift-version 5 -O -I "$(pwd)/build/swift/prefix/lib/swift" -I "$(pwd)/build/swift/prefix/share/swift/CDatamunge" -L "$(pwd)/build/swift/prefix/lib" -lDatamunge -L "$swift_root/lib" -Xlinker -rpath -Xlinker "$(pwd)/build/swift/prefix/lib" -Xlinker -rpath -Xlinker "$swift_root/lib"'
# Compiles a Free Pascal program against build/pascal/prefix (just build-pascal): `uses datamunge` finds the
# compiled unit in lib/fpc/datamunge, which links libdatamunge_pascal (rpath). Append the source; the
# program goes to build/pascal/bin.
FPC := 'mkdir -p build/pascal/bin build/pascal/obj && fpc -O2 -v0ew -l- -Fu"$(pwd)/build/pascal/prefix/lib/fpc/datamunge" -Fl"$(pwd)/build/pascal/prefix/lib" -k-rpath -k"$(pwd)/build/pascal/prefix/lib" -FUbuild/pascal/obj -FEbuild/pascal/bin'
# Runs a Dart program of a package depending on the generated package (tests/dart, examples/dart): resolves
# its dependencies offline first. The native library is DATAMUNGE_DART_LIBRARY (build/dart/prefix, the
# dev shells' hook). Append the package directory and the program.
DART_RUN := 'export DATAMUNGE_DART_LIBRARY="$(pwd)/build/dart/prefix/lib/libdatamunge_dart.so" && dart_run() { (cd "$1" && dart pub get --offline >/dev/null) && dart run "$2"; } && dart_run'
# Builds an Ada program against build/ada/prefix (just build-ada): `with Datamunge` finds the compiled
# package in lib/ada/datamunge (read-only .ali, never recompiled), which links libdatamunge_ada (rpath).
# Append the source directory and the main source; the program goes to build/ada/bin.
GNATMAKE := 'mkdir -p build/ada/bin build/ada/obj && ada_build() { prefix="$(pwd)/build/ada/prefix" && src="$(cd "$1" && pwd)" && main="$(basename "$2" .adb)" && (cd build/ada/obj && gnatmake -q -gnat2022 -aI"$prefix/lib/ada/datamunge" -aO"$prefix/lib/ada/datamunge" -aI"$src" "$src/$main.adb" -o "../bin/$main" -largs -L"$prefix/lib" -Wl,-rpath,"$prefix/lib"); } && ada_build'
# Compiles an Objective-C program against build/objc/prefix (just build-objc): `#import "DM.h"` finds the
# header in include/DM; the program links libDM (the classes), GNUstep Base and libobjc2. Append the source
# and `-o <exe>`.
OBJCC := 'mkdir -p build/objc/bin && objcc() { prefix="$(pwd)/build/objc/prefix" && clang -O2 -fobjc-arc -fobjc-arc-exceptions -fobjc-runtime=gnustep-2.0 -fblocks -Wall -I"$DATAMUNGE_GNUSTEP_BASE_ROOT/include" -I"$DATAMUNGE_LIBOBJC_ROOT/include" -I"$prefix/include/DM" "$@" -L"$prefix/lib" -lDM -L"$DATAMUNGE_GNUSTEP_BASE_LIB/lib" -L"$DATAMUNGE_LIBOBJC_ROOT/lib" -lgnustep-base -lobjc -lm -Wl,-rpath,"$prefix/lib"; } && objcc'
# Runs a SWI-Prolog program with build/prolog/prefix/prolog on the library path (just build-prolog):
# `:- use_module(library(datamunge))` loads the module, which loads ../lib/libdatamunge_prolog. Append the program.
SWIPL := 'swipl -p library="$(pwd)/build/prolog/prefix/prolog"'
# Compiles a Haskell program against the GHC package datamunge registered in build/haskell/prefix (just
# build-haskell); it links libdatamunge_haskell (rpath). Append the source and `-o <exe>`.
GHC := 'mkdir -p build/haskell/bin build/haskell/obj && ghc -v0 -O -threaded -Wall -package-db "$(pwd)/build/haskell/prefix/lib/haskell/package.db" -package datamunge -optl-Wl,-rpath,"$(pwd)/build/haskell/prefix/lib"'
RUST_ENV := 'export DATAMUNGE_RUST_LIB_DIR="$(pwd)/build/rust/prefix/lib" RUSTFLAGS="-C link-arg=-Wl,-rpath,$(pwd)/build/rust/prefix/lib"'
# Runs tclsh with build/tcl/prefix/lib (just build-tcl) on TCLLIBPATH (a Tcl list, so space-separated):
# `package require datamunge` finds the package directory datamunge/ there (pkgIndex.tcl, datamunge.tcl
# and the extension datamunge.so).
TCLSH := 'TCLLIBPATH="$(pwd)/build/tcl/prefix/lib${TCLLIBPATH:+ $TCLLIBPATH}" tclsh'
# octave-cli with build/octave/prefix/share/octave/site/m (just build-octave) on the load path: it holds
# the oct-file __datamunge__.oct and the package +datamunge (datamunge.DataFrame (...)). The path is
# added after silencing Octave:shadowed-function, which Octave raises (on every path refresh) for the
# package functions datamunge.resize and datamunge.Data.load although they shadow nothing. Append
# Octave code and a closing double quote, e.g. {{ OCTAVE }} source examples/octave/x.m".
# guile with build/guile/prefix/share/guile/site/<version> (just build-guile) first on GUILE_LOAD_PATH:
# (use-modules (datamunge)) loads datamunge.scm there, which loads the extension libdatamunge_guile.so
# beside it. GUILE_LOAD_COMPILED_PATH has the datamunge.go that build-guile precompiles, so the module
# loads compiled; auto-compilation stays off (a stale .go falls back to the source, never a 40 s compile).
GUILE := 'GUILE_LOAD_PATH="$(pwd)/build/guile/prefix/share/guile/site/$(pkg-config --variable=effective-version guile-3.0)${GUILE_LOAD_PATH:+:$GUILE_LOAD_PATH}" GUILE_LOAD_COMPILED_PATH="$(pwd)/build/guile/prefix/lib/guile/$(pkg-config --variable=effective-version guile-3.0)/site-ccache${GUILE_LOAD_COMPILED_PATH:+:$GUILE_LOAD_COMPILED_PATH}" guile --no-auto-compile'
# java with build/java/prefix (just build-java): datamunge.jar on the class path and the JNI library
# libdatamunge_java named by DATAMUNGE_JAVA_LIBRARY. Append a source file (single-file launch) or a class.
JAVA := 'DATAMUNGE_JAVA_LIBRARY="$(pwd)/build/java/prefix/lib/libdatamunge_java.so" java -cp "$(pwd)/build/java/prefix/share/java/datamunge.jar"'
# examples/java/<PascalCase>.java for TARGET=snake_case_name (dataframe_ex -> DataframeEx.java).
# go with build/go/prefix (just build-go): cgo links libdatamunge_go from there (with an rpath).
# The workspace build/go/just.go.work (like the dev shells' GOWORK) holds the committed module
# src/datamungego/go/datamunge and the modules examples/go and tests/go (whose go.mod files replace
# datamunge with it, for mep's Go server), so `go run examples/go/x.go` works from the root. Offline.
GO_ENV := 'mkdir -p build/go && printf "go 1.24\n\nuse (\n\t%s\n\t%s\n\t%s\n)\n" "$(pwd)/src/datamungego/go/datamunge" "$(pwd)/examples/go" "$(pwd)/tests/go" > build/go/just.go.work && export GOWORK="$(pwd)/build/go/just.go.work" GOTOOLCHAIN=local GOPROXY=off GOFLAGS= CGO_ENABLED=1 CGO_LDFLAGS="-L$(pwd)/build/go/prefix/lib -Wl,-rpath,$(pwd)/build/go/prefix/lib"'
JAVA_EXAMPLE := 'java_example() { printf "examples/java/%s.java" "$(printf %s "$1" | sed -E "s/(^|_)([a-z0-9])/\\U\\2/g")"; }'
# ocamlfind with build/ocaml/prefix/lib (just build-ocaml) on OCAMLPATH: `-package datamunge` finds the
# findlib package datamunge/ there. ocaml_build SRC EXE compiles SRC natively (a copy under
# build/ocaml/obj, so no .cmi/.cmx/.o land beside it) into EXE, linking libdatamunge_stubs.a and
# libdatamunge (rpath) through datamunge.cmxa.
OCAML_ENV := 'export OCAMLPATH="$(pwd)/build/ocaml/prefix/lib${OCAMLPATH:+:$OCAMLPATH}" CAML_LD_LIBRARY_PATH="$(pwd)/build/ocaml/prefix/lib/datamunge${CAML_LD_LIBRARY_PATH:+:$CAML_LD_LIBRARY_PATH}"'
OCAML_BUILD := 'ocaml_build() { o="build/ocaml/obj/$(basename "$1" .ml)"; mkdir -p "$o" "$(dirname "$2")" && cp "$1" "$o/" && ocamlfind ocamlopt -package datamunge -linkpkg -o "$2" "$o/$(basename "$1")"; }'
# GNU APL with build/apl/prefix (just build-apl): the script datamunge.apl (the dm∆ functions) loads the
# native function library DATAMUNGE_APL_LIBRARY names. Append `-f program.apl`. GNU APL reports errors
# and carries on (exit status 0), so apl_run FILE also fails when the output has an error's caret line
# (`      ^   ^` under the failing statement); --OFF ends the session after the last file.
APL := 'DATAMUNGE_APL_LIBRARY="$(pwd)/build/apl/prefix/lib/libdatamunge_gnu_apl.so" apl --script --noSV --noColor --silent --OFF -f "$(pwd)/build/apl/prefix/share/apl/datamunge.apl"'
APL_RUN := 'apl_run() { mkdir -p build/apl/log && log="build/apl/log/$(basename "$1" .apl).log" && DATAMUNGE_APL_LIBRARY="$(pwd)/build/apl/prefix/lib/libdatamunge_gnu_apl.so" apl --script --noSV --noColor --silent --OFF -f "$(pwd)/build/apl/prefix/share/apl/datamunge.apl" -f "$1" 2>&1 | tee "$log" && ! grep -Eq "^ *\^( +\^)* *$" "$log"; }'
OCTAVE := 'octave-cli --norc --no-history --quiet --eval "warning off Octave:shadowed-function; addpath (\"$(pwd)/build/octave/prefix/share/octave/site/m\");'

# {{{ run commands



run: run-cpp

test: test-cpp

format:
  {{ NIX_DEVELOP }} .#format --command bash -c './scripts/format.sh'

format-check:
  {{ NIX_DEVELOP }} .#quality --command bash -c './scripts/format_check.sh'

lint:
  {{ NIX_DEVELOP }} .#quality --command bash -c './scripts/lint.sh'

quality: format-check lint

fmt: format

run-all: run-cpp run-csharp run-java run-go run-rust run-d run-python run-php run-perl run-tcl run-lua run-ruby run-r run-guile run-javascript run-ocaml run-octave run-apl

# Each example is a .NET 10 file-based app: `#:project ../../src/datamungecsharp/Datamunge.csproj`
# references the generated project, and the dev shells point DATAMUNGE_CSHARP_LIBRARY at the
# native library build-csharp installs.
run-csharp: build-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'dotnet run examples/csharp/{{ TARGET }}.cs'

# Visual Basic has no file-based apps: examples/vb/Example.vbproj builds the one example
# examples/vb/TARGET.vb against the generated project src/datamungevb/Datamunge.vbproj, and the dev
# shells point DATAMUNGE_VB_LIBRARY at the native library build-vb installs.
run-vb: build-vb
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'dotnet run --project examples/vb/Example.vbproj --property:Example={{ TARGET }}'

# Runs examples/java/<Target>.java (TARGET=dataframe_ex runs DataframeEx.java) as a single-file
# source launch against build/java/prefix (Rosetta binding).
run-java: build-java
  {{ NIX_DEVELOP }} .#java --command bash -c '{{ JAVA_EXAMPLE }}; {{ JAVA }} "$(java_example {{ TARGET }})"'

# Runs examples/go/TARGET.go (a `//go:build ignore` main program in the module examples/go) with
# `go run` from the repository root against build/go/prefix (Rosetta binding). Go has no REPL, so
# there is no repl-go.
run-go: build-go
  {{ NIX_DEVELOP }} .#go --command bash -c '{{ GO_ENV }} && go run examples/go/{{ TARGET }}.go'

# Runs examples/apl/TARGET.apl (GNU APL) from the repository root against build/apl/prefix (Rosetta
# binding); fails when GNU APL reports an error.
run-apl: build-apl
  {{ NIX_DEVELOP }} .#apl --command bash -c 'set -o pipefail; {{ APL_RUN }}; apl_run examples/apl/{{ TARGET }}.apl'

# Each example is examples/rust/TARGET.rs, a Cargo example of the crate examples/rust built against the
# Rosetta crate src/datamungerust/datamunge; RUST_ENV points it at libdatamunge_rust (just build-rust).
run-rust: build-rust
  {{ NIX_DEVELOP }} .#rust --command bash -c '{{ RUST_ENV }} && cargo run --quiet --release --manifest-path examples/rust/Cargo.toml --example {{ TARGET }}'

# Compiles examples/haskell/TARGET.hs with GHC into build/haskell/bin and runs it.
run-haskell: build-haskell
  {{ NIX_DEVELOP }} .#haskell --command bash -c '{{ GHC }} -outputdir build/haskell/obj/{{ TARGET }} examples/haskell/{{ TARGET }}.hs -o build/haskell/bin/{{ TARGET }} && build/haskell/bin/{{ TARGET }}'

# Runs examples/prolog/TARGET.pl against build/prolog/prefix.
run-prolog: build-prolog
  {{ NIX_DEVELOP }} .#prolog --command bash -c '{{ SWIPL }} examples/prolog/{{ TARGET }}.pl'

# Compiles examples/objc/TARGET.m with OBJCC into build/objc/bin and runs it.
run-objc: build-objc
  {{ NIX_DEVELOP }} .#objc --command bash -c '{{ OBJCC }} examples/objc/{{ TARGET }}.m -o build/objc/bin/{{ TARGET }} && build/objc/bin/{{ TARGET }}'

# Builds examples/ada/TARGET.adb with GNATMAKE into build/ada/bin and runs it.
run-ada: build-ada
  {{ NIX_DEVELOP }} .#ada --command bash -c '{{ GNATMAKE }} examples/ada examples/ada/{{ TARGET }}.adb && build/ada/bin/{{ TARGET }}'

# Runs examples/dart/TARGET.dart against build/dart/prefix.
run-dart: build-dart
  {{ NIX_DEVELOP }} .#dart --command bash -c '{{ DART_RUN }} examples/dart examples/dart/{{ TARGET }}.dart'

# Compiles examples/pascal/TARGET.pas with FPC into build/pascal/bin and runs it.
run-pascal: build-pascal
  {{ NIX_DEVELOP }} .#pascal --command bash -c '{{ FPC }} examples/pascal/{{ TARGET }}.pas && build/pascal/bin/{{ TARGET }}'

# Compiles examples/swift/TARGET.swift with SWIFTC into build/swift/bin and runs it.
run-swift: build-swift
  {{ NIX_DEVELOP }} .#swift --command bash -c '{{ SWIFTC }} examples/swift/{{ TARGET }}.swift -o build/swift/bin/{{ TARGET }} && build/swift/bin/{{ TARGET }}'

run-d: build-d
  bash -c 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    cp examples/d/examples-src/{{ TARGET }}.d examples/d/source/app.d; \
    if command -v nix >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -c "rm -rf build/dub-packages/datamunged-0.0.1 && cd examples/d && dub run $compiler --build=release"; \
    else \
      rm -rf build/dub-packages/datamunged-0.0.1 && cd examples/d && dub run $compiler --build=release; \
    fi'

run-python: build-python
  {{ NIX_DEVELOP }} .#python --command bash -c 'env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/{{ TARGET }}.py'

# Each example requires the generated src/datamungephp/php/datamunge.php, which loads the library
# build-php installs from DATAMUNGE_PHP_LIBRARY (set by the dev shells).
run-php: build-php
  {{ NIX_DEVELOP }} .#php --command bash -c 'php examples/php/{{ TARGET }}.php'

# Runs examples/perl/TARGET.pl with build/perl/prefix/lib/perl5 (just build-perl: Datamunge.pm and
# auto/Datamunge/libdatamunge_perl.so) on PERL5LIB.
run-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -c 'export PERL5LIB="$(pwd)/build/perl/prefix/lib/perl5${PERL5LIB:+:$PERL5LIB}" && perl examples/perl/{{ TARGET }}.pl'

# Runs examples/tcl/TARGET.tcl from the repository root against build/tcl/prefix (Rosetta binding).
run-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -c '{{ TCLSH }} examples/tcl/{{ TARGET }}.tcl'

run-lua: build-lua
  {{ NIX_DEVELOP }} .#lua --command bash -c 'cmake --install build/datamungelua --prefix build/lua/prefix >/dev/null && export LUA_CPATH="$(pwd)/build/lua/prefix/lib/lua/?.so;$(pwd)/build/lua/prefix/lib64/lua/?.so;;" && export LUA_PATH="$(pwd)/build/lua/prefix/share/lua/?.lua;;" && export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && export DYLD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${DYLD_LIBRARY_PATH:+:}$DYLD_LIBRARY_PATH" && lua examples/lua/{{ TARGET }}.lua'

run-lisp: build-lisp
  {{ NIX_DEVELOP }} .#lisp --command bash -c 'sbcl --noinform --non-interactive --load examples/lisp/{{ TARGET }}.lisp'

run-julia: build-julia
  {{ NIX_DEVELOP }} .#julia --command bash -c 'export JULIA_LOAD_PATH="$(pwd)/build/julia/prefix/share/julia:@:@v#.#:@stdlib" && julia --startup-file=no --threads=auto examples/julia/{{ TARGET }}.jl'

# Compiles examples/fortran/TARGET.f90 against build/fortran/prefix into build/fortran/bin and runs it
# from the repository root (Fortran has no REPL, so there is no repl-fortran).
run-fortran: build-fortran
  {{ NIX_DEVELOP }} .#fortran --command bash -c '{{ FORTRAN_FC }} examples/fortran/{{ TARGET }}.f90 {{ FORTRAN_LIBS }} -o build/fortran/bin/{{ TARGET }} && build/fortran/bin/{{ TARGET }}'

run-cobol: build-cobol
  {{ NIX_DEVELOP }} .#cobol --command bash -c 'mkdir -p build/cobol/bin && {{ COBC }} -o build/cobol/bin/{{ TARGET }} examples/cobol/{{ TARGET }}.cob && build/cobol/bin/{{ TARGET }}'

# Runs examples/ruby/TARGET.rb from the repository root against build/ruby/prefix (Rosetta binding).
run-ruby: build-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -c '{{ RUBY }} examples/ruby/{{ TARGET }}.rb'

run-r: build-r
   {{ NIX_DEVELOP }} .#r --command bash -c 'R_LIBS_USER="$(pwd)/build/r/library${R_LIBS_USER:+:}$R_LIBS_USER" Rscript examples/r/{{ TARGET }}.r'

# Runs examples/guile/TARGET.scm from the repository root against build/guile/prefix (Rosetta binding).
run-guile: build-guile
  {{ NIX_DEVELOP }} .#guile --command bash -c '{{ GUILE }} -s examples/guile/{{ TARGET }}.scm'

run-javascript: build-javascript
   {{ NIX_DEVELOP }} .#javascript --command bash -c 'node ./examples/javascript/{{ TARGET }}.js'

# Compiles examples/ocaml/TARGET.ml natively against build/ocaml/prefix (Rosetta binding) and runs it
# from the repository root.
run-ocaml: build-ocaml
  {{ NIX_DEVELOP }} .#ocaml --command bash -c '{{ OCAML_ENV }} && {{ OCAML_BUILD }}; ocaml_build examples/ocaml/{{ TARGET }}.ml build/ocaml/bin/{{ TARGET }} && ./build/ocaml/bin/{{ TARGET }}'

# Runs examples/octave/TARGET.m from the repository root against build/octave/prefix (Rosetta binding).
run-octave: build-octave
  {{ NIX_DEVELOP }} .#octave --command bash -c '{{ OCTAVE }} source examples/octave/{{ TARGET }}.m"'

# run-all-<lang>: build the binding once, then run EVERY example for that language,
# printing a per-example header and a passed/failed summary (exits non-zero on any failure).

run-all-lua: build-lua
  {{ NIX_DEVELOP }} .#lua --command bash -c 'cmake --install build/datamungelua --prefix build/lua/prefix >/dev/null && \
    export LUA_CPATH="$(pwd)/build/lua/prefix/lib/lua/?.so;$(pwd)/build/lua/prefix/lib64/lua/?.so;;" && \
    export LUA_PATH="$(pwd)/build/lua/prefix/share/lua/?.lua;;" && \
    export LD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH" && \
    export DYLD_LIBRARY_PATH="$(pkg-config --variable=libdir datamunge)${DYLD_LIBRARY_PATH:+:}$DYLD_LIBRARY_PATH" && \
    pass=0; fail=0; failed=""; \
    for f in examples/lua/*.lua; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if lua "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## lua: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-lisp: build-lisp
  {{ NIX_DEVELOP }} .#lisp --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/lisp/*.lisp; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if sbcl --noinform --non-interactive --load "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## lisp: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-csharp: build-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/csharp/*.cs; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if dotnet run "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## csharp: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-vb: build-vb
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/vb/*_ex.vb; do n="$(basename "$f" .vb)"; printf "\n========== %s ==========\n" "$n.vb"; \
      if dotnet run --project examples/vb/Example.vbproj --property:Example="$n"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n.vb"; fi; done; \
    printf "\n########## vb: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-rust: build-rust
  {{ NIX_DEVELOP }} .#rust --command bash -c '{{ RUST_ENV }} && cargo build --quiet --release --manifest-path examples/rust/Cargo.toml --examples && pass=0; fail=0; failed=""; \
    for f in examples/rust/*_ex.rs; do n="$(basename "$f" .rs)"; printf "\n========== %s ==========\n" "$n.rs"; \
      if cargo run --quiet --release --manifest-path examples/rust/Cargo.toml --example "$n"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n.rs"; fi; done; \
    printf "\n########## rust: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-haskell: build-haskell
  {{ NIX_DEVELOP }} .#haskell --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/haskell/*_ex.hs; do n="$(basename "$f" .hs)"; printf "\n========== %s ==========\n" "$n.hs"; \
      if {{ GHC }} -outputdir "build/haskell/obj/$n" "$f" -o "build/haskell/bin/$n" && build/haskell/bin/$n; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n.hs"; fi; done; \
    printf "\n########## haskell: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-prolog: build-prolog
  {{ NIX_DEVELOP }} .#prolog --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/prolog/*_ex.pl; do n="$(basename "$f" .pl)"; printf "\n========== %s ==========\n" "$n.pl"; \
      if {{ SWIPL }} "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n.pl"; fi; done; \
    printf "\n########## prolog: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-objc: build-objc
  {{ NIX_DEVELOP }} .#objc --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/objc/*_ex.m; do n="$(basename "$f" .m)"; printf "\n========== %s ==========\n" "$n.m"; \
      if {{ OBJCC }} "$f" -o build/objc/bin/$n && build/objc/bin/$n; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n.m"; fi; done; \
    printf "\n########## objc: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-ada: build-ada
  {{ NIX_DEVELOP }} .#ada --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/ada/*_ex.adb; do n="$(basename "$f" .adb)"; printf "\n========== %s ==========\n" "$n.adb"; \
      if {{ GNATMAKE }} examples/ada "$f" && build/ada/bin/$n; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n.adb"; fi; done; \
    printf "\n########## ada: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-dart: build-dart
  {{ NIX_DEVELOP }} .#dart --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/dart/*_ex.dart; do n="$(basename "$f" .dart)"; printf "\n========== %s ==========\n" "$n.dart"; \
      if {{ DART_RUN }} examples/dart "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n.dart"; fi; done; \
    printf "\n########## dart: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-pascal: build-pascal
  {{ NIX_DEVELOP }} .#pascal --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/pascal/*_ex.pas; do n="$(basename "$f" .pas)"; printf "\n========== %s ==========\n" "$n.pas"; \
      if {{ FPC }} "$f" && "build/pascal/bin/$n"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n.pas"; fi; done; \
    printf "\n########## pascal: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-php: build-php
  {{ NIX_DEVELOP }} .#php --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/php/*_ex.php; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if php "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## php: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-swift: build-swift
  {{ NIX_DEVELOP }} .#swift --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/swift/*_ex.swift; do n="$(basename "$f" .swift)"; printf "\n========== %s ==========\n" "$n.swift"; \
      if {{ SWIFTC }} "$f" -o "build/swift/bin/$n" && "build/swift/bin/$n"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n.swift"; fi; done; \
    printf "\n########## swift: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-julia: build-julia
  {{ NIX_DEVELOP }} .#julia --command bash -c 'export JULIA_LOAD_PATH="$(pwd)/build/julia/prefix/share/julia:@:@v#.#:@stdlib" && pass=0; fail=0; failed=""; \
    for f in examples/julia/*.jl; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if julia --startup-file=no --threads=auto "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## julia: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-fortran: build-fortran
  {{ NIX_DEVELOP }} .#fortran --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/fortran/*.f90; do n="$(basename "$f" .f90)"; printf "\n========== %s ==========\n" "$n"; \
      if {{ FORTRAN_FC }} "$f" {{ FORTRAN_LIBS }} -o "build/fortran/bin/$n" && "build/fortran/bin/$n"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## fortran: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-cobol: build-cobol
  {{ NIX_DEVELOP }} .#cobol --command bash -c 'mkdir -p build/cobol/bin; pass=0; fail=0; failed=""; \
    for f in examples/cobol/*.cob; do n="$(basename "$f" .cob)"; printf "\n========== %s ==========\n" "$n"; \
      if {{ COBC }} -o "build/cobol/bin/$n" "$f" && "build/cobol/bin/$n"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## cobol: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-guile: build-guile
  {{ NIX_DEVELOP }} .#guile --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/guile/*.scm; do n="$(basename "$f" .scm)"; printf "\n========== %s ==========\n" "$n"; \
      if {{ GUILE }} -s "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## guile: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-ruby: build-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/ruby/*.rb; do n="$(basename "$f" .rb)"; printf "\n========== %s ==========\n" "$n"; \
      if {{ RUBY }} "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## ruby: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-ocaml: build-ocaml
  {{ NIX_DEVELOP }} .#ocaml --command bash -c '{{ OCAML_ENV }} && {{ OCAML_BUILD }}; pass=0; fail=0; failed=""; \
    for f in examples/ocaml/*.ml; do n="$(basename "$f" .ml)"; printf "\n========== %s ==========\n" "$n"; \
      if ocaml_build "$f" "build/ocaml/bin/$n" && "./build/ocaml/bin/$n"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## ocaml: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/tcl/*.tcl; do n="$(basename "$f" .tcl)"; printf "\n========== %s ==========\n" "$n"; \
      if {{ TCLSH }} "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## tcl: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-octave: build-octave
  {{ NIX_DEVELOP }} .#octave --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/octave/*.m; do n="$(basename "$f" .m)"; printf "\n========== %s ==========\n" "$n"; \
      if {{ OCTAVE }} source $f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## octave: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -c 'export PERL5LIB="$(pwd)/build/perl/prefix/lib/perl5${PERL5LIB:+:$PERL5LIB}" && \
    pass=0; fail=0; failed=""; \
    for f in examples/perl/*_ex.pl; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if perl "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## perl: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-go: build-go
  {{ NIX_DEVELOP }} .#go --command bash -c '{{ GO_ENV }} && pass=0; fail=0; failed=""; \
    for f in examples/go/*.go; do n="$(basename "$f" .go)"; printf "\n========== %s ==========\n" "$n"; \
      if go run "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## go: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-apl: build-apl
  {{ NIX_DEVELOP }} .#apl --command bash -c 'set -o pipefail; {{ APL_RUN }}; pass=0; fail=0; failed=""; \
    for f in examples/apl/*.apl; do n="$(basename "$f" .apl)"; printf "\n========== %s ==========\n" "$n"; \
      if apl_run "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## apl: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-d: build-d
  {{ NIX_DEVELOP }} .#d --command bash -c 'compiler=""; if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    pass=0; fail=0; failed=""; \
    for f in examples/d/examples-src/*.d; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      cp "$f" examples/d/source/app.d; \
      if ( rm -rf build/dub-packages/datamunged-0.0.1 && cd examples/d && dub run $compiler --build=release ); then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## d: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-python: build-python
  {{ NIX_DEVELOP }} .#python --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/python/*.py; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if build/venv/pydatamunge-run/bin/python "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## python: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-r: build-r
  {{ NIX_DEVELOP }} .#r --command bash -c 'export R_LIBS_USER="$(pwd)/build/r/library${R_LIBS_USER:+:}$R_LIBS_USER"; \
    pass=0; fail=0; failed=""; \
    for f in examples/r/*.r; do n="$(basename "$f")"; printf "\n========== %s ==========\n" "$n"; \
      if Rscript "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## r: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-java: build-java
  {{ NIX_DEVELOP }} .#java --command bash -c 'pass=0; fail=0; failed=""; \
    for f in examples/java/*.java; do \
      n="$(basename "$f" .java | sed -E "s/([a-z0-9])([A-Z])/\1_\2/g" | tr "[:upper:]" "[:lower:]")"; \
      printf "\n========== %s ==========\n" "$n"; \
      if {{ JAVA }} "$f"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done; \
    printf "\n########## java: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

run-all-cpp: examples
  {{ NIX_DEVELOP }} .#cpp --command bash -c 'pass=0; fail=0; failed=""; \
    while IFS= read -r f; do n="$(basename "$f" .cpp)"; b="build/debug/examples/$n"; \
      printf "\n========== %s ==========\n" "$n"; \
      if [ -x "$b" ] && "$b"; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $n"; fi; done < <(find examples/cpp/source -path examples/cpp/source/nn -prune -o -type f -name "*.cpp" -print); \
    printf "\n########## cpp: %d passed, %d failed ##########\n" "$pass" "$fail"; \
    if [ -n "$failed" ]; then echo "FAILED:$failed"; fi; [ "$fail" -eq 0 ]'

## Build and run only the selected working-tree example; forward optional program arguments.
[positional-arguments]
run-cpp *ARGS:
    @echo "Running target {{ TARGET }}"
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S examples/cpp -B build/debug/examples -DCMAKE_BUILD_TYPE=Debug -DBUILD_W_INSTALLED=OFF
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/debug/examples --target {{ TARGET }} --parallel {{ JOBS }}
    {{ NIX_DEVELOP }} .#cpp-tools --command ./build/debug/examples/{{ TARGET }} "$@"

run-plot:
    @echo "Running plot example"
    {{ NIX_DEVELOP }} .#cpp-tools --command bash -c 'cmake -S examples/cpp -B build/debug/examples --preset=debug -DCMAKE_MAKE_PROGRAM="$(command -v make)" -DBUILD_W_INSTALLED=OFF'
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/debug/examples --target plot_ex --parallel {{ JOBS }}
    ./build/debug/examples/plot_ex

view-plot: examples
    @echo "Running plot example and opening the scatter plot"
    {{ NIX_DEVELOP }} .#cpp --command bash -c './build/debug/examples/plot_ex'

run-benchmark:
    @echo "Running Benchmarks"
    python3 scripts/benchmark_backend.py run --no-build --suite {{ BENCH_TARGET }}


# }}} run commands

# {{{ prebuild commands

# Regenerate the embedded Computer Modern plot font (src/datamunge/plot/cmu_serif_font_data.hpp)
# from CMU Serif + Latin Modern Math, using the flake's pinned nixpkgs.
embed-plot-font:
  #!/usr/bin/env bash
  set -euo pipefail
  mapfile -t deps < <(nix build --inputs-from . --no-link --print-out-paths \
    nixpkgs#texlivePackages.cm-unicode.tex nixpkgs#texlivePackages.lm-math.tex \
    nixpkgs#python3 nixpkgs#python3Packages.fonttools)
  PYTHONPATH="$(echo "${deps[3]}"/lib/python3*/site-packages)" "${deps[2]}/bin/python3" scripts/embed_plot_font.py \
    --font "${deps[0]}/fonts/opentype/public/cm-unicode/cmunrm.otf" \
    --fallback "${deps[1]}/fonts/opentype/public/lm-math/latinmodern-math.otf"

# Update the pinned swig fork revision in flake.lock to the latest github:jordanschupbach/swig.
update-swig:
  nix flake update swig

prebuild-swig: prebuild-python prebuild-javascript prebuild-r prebuild-lua prebuild-d prebuild-octave
  @echo "SWIG wrappers regenerated (with Doxygen comments enabled)"

prebuild-python:
  {{ NIX_DEVELOP }} .#python --command python scripts/prebuild_bindings.py python

# JavaScript (Rosetta): src/datamunge_js_wrap*.cpp (the Node-API addon) and index.d.ts, whose JSDoc
# carries the C++ documentation for editors (tsserver, mep's K).
prebuild-javascript:
  {{ NIX_DEVELOP }} .#javascript --command python3 scripts/prebuild_bindings.py javascript

# C# (Rosetta): src/datamunge_csharp_wrap*.cpp (the native library datamunge_csharp) and the project
# src/datamungecsharp (Datamunge.cs, Datamunge.csproj), whose XML doc comments carry the Doxygen documentation.
prebuild-csharp:
  {{ NIX_DEVELOP }} .#csharp --command python3 scripts/prebuild_bindings.py csharp

# Visual Basic (Rosetta): src/datamunge_vb_wrap*.cpp (the native library datamunge_vb) and the project
# src/datamungevb (Datamunge.vb, Datamunge.vbproj), whose XML doc comments carry the Doxygen documentation.
prebuild-vb:
  {{ NIX_DEVELOP }} .#csharp --command python3 scripts/prebuild_bindings.py vb

# Fortran (Rosetta): src/datamunge_fortran_wrap*.cpp (the native library libdatamunge_fortran) and the
# Fortran 2018 module source src/datamungefortran/datamunge.f90, whose FORD `!>` comments carry the
# Doxygen documentation (mep's Fortran LSP reads them from this committed copy).
prebuild-fortran:
  {{ NIX_DEVELOP }} .#fortran --command python3 scripts/prebuild_bindings.py fortran

prebuild-r:
  {{ NIX_DEVELOP }} .#r --command python3 scripts/prebuild_bindings.py r

# Perl (Rosetta): src/datamunge_perl_wrap*.cpp (the library libdatamunge_perl with its XS glue) and the
# module src/datamungeperl/lib/Datamunge.pm, whose POD carries the Doxygen documentation.
prebuild-perl:
  {{ NIX_DEVELOP }} .#perl --command python3 scripts/prebuild_bindings.py perl

# Ruby (Rosetta): src/datamunge_ruby_wrap*.cpp (the MRI extension datamunge/datamunge.so) and
# src/datamungeruby/lib/datamunge.rb, which loads it and whose YARD comments carry the Doxygen
# documentation (yard, mep's Ruby LSP read this committed copy).
prebuild-ruby:
  {{ NIX_DEVELOP }} .#ruby --command python3 scripts/prebuild_bindings.py ruby

# Tcl (Rosetta): src/datamunge_tcl_wrap*.cpp (the stubs extension datamunge.so, Datamunge_Init) and
# the package files src/datamungetcl/pkg/datamunge/{datamunge.tcl,pkgIndex.tcl}; datamunge.tcl loads
# the extension and its commented proc/oo::class stubs carry the Doxygen documentation (mep's Tcl
# LSP reads this committed copy through MEP_TCL_PATH).
prebuild-tcl:
  {{ NIX_DEVELOP }} .#tcl --command python3 scripts/prebuild_bindings.py tcl

# Lua (Rosetta): src/datamunge_lua_wrap*.cpp and the LuaCATS stub src/datamungelua/library/datamunge.lua,
# which .luarc.json points editors (lua-language-server, mep) at for hover and completion.
prebuild-lua:
  {{ NIX_DEVELOP }} .#lua --command python3 scripts/prebuild_bindings.py lua

# Common Lisp (Rosetta): src/datamunge_lisp_wrap*.cpp and src/datamungelisp/datamunge.{lisp,asd}
prebuild-lisp:
  {{ NIX_DEVELOP }} .#lisp --command python3 scripts/prebuild_bindings.py lisp

# Julia (Rosetta): src/datamunge_julia_wrap*.cpp and the package src/datamungejulia
# (Project.toml, src/Datamunge.jl), whose docstrings carry the Doxygen documentation.
prebuild-julia:
  {{ NIX_DEVELOP }} .#julia --command python3 scripts/prebuild_bindings.py julia

# COBOL (Rosetta): src/datamunge_cobol_wrap*.cpp and the copybooks
# src/datamungecobol/copybooks/datamunge{,-callbacks}.cpy (record prefix DM)
prebuild-cobol:
  {{ NIX_DEVELOP }} .#cobol --command python3 scripts/prebuild_bindings.py cobol

# D (SWIG)
prebuild-d:
  bash -c 'set -euo pipefail; \
    cmd="mkdir -p {{ BINDINGS_DIR }}/datamunged/source && swig -c++ -d -Iinclude -o {{ BINDINGS_DIR }}/datamunged/source/datamunged_wrap.cpp -oh {{ BINDINGS_DIR }}/datamunged/source/datamunged_wrap.h -outdir {{ BINDINGS_DIR }}/datamunged/source src/datamunged/swig/datamunged.i"; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -c "$cmd"; \
    else \
      bash -c "$cmd"; \
    fi'

# Guile (SWIG)
# Guile (Rosetta): src/datamunge_guile_wrap*.cpp (the extension libdatamunge_guile) and the modules
# src/datamungeguile/scheme/datamunge.scm and datamunge/*.scm, whose docstrings carry the Doxygen
# documentation (procedure-documentation, mep's Scheme LSP through MEP_GUILE_PATH). Stale .scm
# files are removed.
prebuild-guile:
  {{ NIX_DEVELOP }} .#guile --command python3 scripts/prebuild_bindings.py guile

# Octave (Rosetta): src/datamunge_octave_wrap*.cpp (the oct-file __datamunge__.oct) and the package
# src/datamungeoctave/inst/+datamunge, whose .m help text carries the Doxygen documentation (`help
# datamunge.DataFrame`, mep's Octave LSP through MEP_OCTAVE_PATH). Stale .m files are removed.
prebuild-octave:
  {{ NIX_DEVELOP }} .#octave --command python3 scripts/prebuild_bindings.py octave

# Go (Rosetta): src/datamunge_go_wrap*.cpp (the shared library libdatamunge_go that cgo links) and
# the module datamunge under src/datamungego/go/datamunge (go.mod, datamunge.go, data/,
# internal/rosetta); its godoc comments carry the C++ documentation (mep's Go server reads this
# committed copy through the go.mod replace of examples/go and tests/go).
prebuild-go:
  {{ NIX_DEVELOP }} .#go --command python3 scripts/prebuild_bindings.py go

# PHP (Rosetta): src/datamunge_php_wrap*.cpp (the shared library libdatamunge_php) and
# src/datamungephp/php/datamunge.php (namespace Datamunge over FFI), whose PHPDoc carries the Doxygen documentation.
prebuild-php:
  {{ NIX_DEVELOP }} .#php --command python3 scripts/prebuild_bindings.py php

# Java (Rosetta): src/datamunge_java_wrap*.cpp (the JNI library libdatamunge_java) and the package
# datamunge under src/datamungejava/java (one .java file per class); its Javadoc carries the C++
# documentation (mep's Java server reads this committed copy through MEP_JAVA_SOURCEPATH).
prebuild-java:
  {{ NIX_DEVELOP }} .#java --command python3 scripts/prebuild_bindings.py java

# OCaml (Rosetta): src/datamunge_ocaml_wrap*.cpp (the C ABI library with its OCaml C stubs) and the
# findlib package src/datamungeocaml/ocaml/datamunge (datamunge.ml, META and datamunge.mli, whose odoc
# comments carry the C++ documentation; mep's OCaml server reads this committed copy through
# MEP_OCAML_PATH).
prebuild-ocaml:
  {{ NIX_DEVELOP }} .#ocaml --command python3 scripts/prebuild_bindings.py ocaml

# GNU APL (Rosetta): src/datamunge_apl_wrap*.cpp (the native function library libdatamunge_gnu_apl) and
# the script src/datamungeapl/apl/datamunge.apl (the dm∆ functions), whose ⍝ comments carry the C++
# documentation (mep's APL server reads this committed copy through MEP_APL_PATH).
prebuild-apl:
  {{ NIX_DEVELOP }} .#apl --command python3 scripts/prebuild_bindings.py apl

# }}} prebuild commands

# Rust (Rosetta): src/datamunge_rust_wrap*.cpp (the shared library libdatamunge_rust) and the Cargo crate
# src/datamungerust/datamunge, whose rustdoc comments carry the Doxygen documentation.
prebuild-rust:
  {{ NIX_DEVELOP }} .#rust --command python3 scripts/prebuild_bindings.py rust

# Haskell (Rosetta): src/datamunge_haskell_wrap*.cpp (the shared library libdatamunge_haskell) and the GHC modules
# src/datamungehaskell/haskell (Datamunge.hs with the Haddock documentation, Datamunge/Rosetta.hs).
prebuild-haskell:
  {{ NIX_DEVELOP }} .#haskell --command python3 scripts/prebuild_bindings.py haskell

# SWI-Prolog (Rosetta): src/datamunge_prolog_wrap*.cpp (the foreign library libdatamunge_prolog) and the module
# src/datamungeprolog/prolog/datamunge.pl, whose PlDoc carries the Doxygen documentation.
prebuild-prolog:
  {{ NIX_DEVELOP }} .#prolog --command python3 scripts/prebuild_bindings.py prolog

# Objective-C (Rosetta): src/datamunge_objc_wrap*.cpp (the shared library libdatamunge_objc) and the classes
# src/datamungeobjc/objc (DM.h with the /// documentation, DM.m with the runtime).
prebuild-objc:
  {{ NIX_DEVELOP }} .#objc --command python3 scripts/prebuild_bindings.py objc

# Ada (Rosetta): src/datamunge_ada_wrap*.cpp (the shared library libdatamunge_ada) and the package
# src/datamungeada/ada (datamunge.ads with the GNATdoc documentation, .adb, .gpr).
prebuild-ada:
  {{ NIX_DEVELOP }} .#ada --command python3 scripts/prebuild_bindings.py ada

# Dart (Rosetta): src/datamunge_dart_wrap*.cpp (the shared library libdatamunge_dart) and the package
# src/datamungedart/datamunge (dart:ffi), whose dartdoc comments carry the Doxygen documentation.
prebuild-dart:
  {{ NIX_DEVELOP }} .#dart --command python3 scripts/prebuild_bindings.py dart

# Free Pascal (Rosetta): src/datamunge_pascal_wrap*.cpp (the shared library libdatamunge_pascal) and the unit
# src/datamungepascal/pascal/datamunge.pas, whose XML doc comments carry the Doxygen documentation.
prebuild-pascal:
  {{ NIX_DEVELOP }} .#pascal --command python3 scripts/prebuild_bindings.py pascal

# Swift (Rosetta): src/datamunge_swift_wrap*.cpp (the shared library libdatamunge_swift) and the Swift
# package src/datamungeswift/Datamunge, whose DocC comments carry the Doxygen documentation.
prebuild-swift:
  {{ NIX_DEVELOP }} .#swift --command python3 scripts/prebuild_bindings.py swift

# {{{ build commands

# Installs libdatamunge_php into build/php/prefix/lib (DATAMUNGE_PHP_LIBRARY in the dev shells).
build-php: prebuild-php
  {{ NIX_DEVELOP }} .#php --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungephp -B build/datamungephp -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#php --command bash -c 'cmake --build build/datamungephp -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#php --command bash -c 'rm -rf build/php/prefix && cmake --install build/datamungephp --prefix build/php/prefix >/dev/null'


build: build-cpp

build-example-installed:
  {{ NIX_DEVELOP }} .#cpp --command bash -c 'cmake -S examples/cpp -B build/debug/examples-installed -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=$(command -v make) -DBUILD_W_INSTALLED=ON'


build-release:
  {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S . -B build/release --preset=release-clang-linux-x86 -DCMAKE_MAKE_PROGRAM=$(command -v make)"
  ln -sf build/release/compile_commands.json compile_commands.json

build-debug:
  {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S . -B build/debug --preset=debug-clang-linux-x86 -DCMAKE_MAKE_PROGRAM=$(command -v make)"
  ln -sf build/debug/compile_commands.json compile_commands.json

build-cpp:
    @echo "Building datamunge"
    {{ NIX_DEVELOP }} .#cpp-tools --command bash -c "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
    {{ NIX_DEVELOP }} .#cpp-tools --command bash -c "cmake --build build --config Release -j{{ JOBS }} --verbose"
    {{ NIX_DEVELOP }} .#cpp-tools --command bash -c "find ./build -name 'compile_commands.json' -exec cat {} + | jq -s add > compile_commands.json"


# Installs the native library and the project into build/csharp/prefix/share/dotnet/Datamunge, and
# builds Datamunge.dll (with Datamunge.xml) into its bin/ for `#r` in a C# console.
build-csharp: prebuild-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungecsharp -B build/datamungecsharp -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'cmake --build build/datamungecsharp -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'rm -rf build/csharp/prefix && cmake --install build/datamungecsharp --prefix build/csharp/prefix'
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'dotnet build src/datamungecsharp/Datamunge.csproj -c Release -nologo -v q -o build/csharp/prefix/share/dotnet/Datamunge/bin'

# Installs the native library and the project into build/vb/prefix/share/dotnet/DatamungeVB, and
# builds Datamunge.dll (with Datamunge.xml) into its bin/.
build-vb: prebuild-vb
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungevb -B build/datamungevb -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'cmake --build build/datamungevb -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'rm -rf build/vb/prefix && cmake --install build/datamungevb --prefix build/vb/prefix'
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'dotnet build src/datamungevb/Datamunge.vbproj -c Release -nologo -v q -o build/vb/prefix/share/dotnet/DatamungeVB/bin'



# Installs libdatamunge_fortran.so (the native library), libdatamunge_fortran_mod.a and the .mod files
# into build/fortran/prefix: compile programs with FORTRAN_FC (run-fortran, test-fortran).
build-fortran: prebuild-fortran
  {{ NIX_DEVELOP }} .#fortran --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungefortran -B build/datamungefortran -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#fortran --command bash -c 'cmake --build build/datamungefortran -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#fortran --command bash -c 'rm -rf build/fortran/prefix && cmake --install build/datamungefortran --prefix build/fortran/prefix'

# Links the shell's datamunge library rather than recompiling the core into the addon.
build-javascript: prebuild-javascript
    {{ NIX_DEVELOP }} .#javascript --command bash -c 'export npm_config_nodedir="$(dirname "$(dirname "$(command -v node)")")" && DATAMUNGE_USE_SYSTEM_LIB=1 npm --prefix . run build'

build-python: prebuild-python
  # Recreate the venv when its interpreter is gone (a nix garbage collection removes the shell's python).
  {{ NIX_DEVELOP }} .#python --command bash -c 'build/venv/pydatamunge-run/bin/python -c "" 2>/dev/null || python -m venv --clear --system-site-packages build/venv/pydatamunge-run'
  # Root the interpreter and the linked datamunge library so nix garbage collection keeps them while the venv uses them.
  {{ NIX_DEVELOP }} .#python --command bash -c 'mkdir -p build/venv/gcroots && rm -f build/venv/gcroots/* && for p in "$(python -c "import sys; print(sys.prefix)")" "$(pkg-config --variable=prefix datamunge)"; do nix-store --add-root "build/venv/gcroots/$(basename "$p")" --indirect --realise "$p" > /dev/null; done'
  # .venv lets editors (mep's Python hover/completion) find the interpreter outside the dev shell.
  test -e .venv || ln -s build/venv/pydatamunge-run .venv
  # Link the shell's datamunge library (built with native data assets) rather than recompiling the core without them.
  {{ NIX_DEVELOP }} .#python --command bash -c 'if test -f build/nn-python/enabled; then export DATAMUNGE_BUILD_NN_EXAMPLES=${DATAMUNGE_BUILD_NN_EXAMPLES:-1}; fi; DATAMUNGE_USE_SYSTEM_LIB=1 CMAKE_BUILD_PARALLEL_LEVEL={{ JOBS }} build/venv/pydatamunge-run/bin/python -m pip install --force-reinstall . --no-deps --no-build-isolation'


# Installs datamunge.jar (share/java) and the JNI library libdatamunge_java (lib) into
# build/java/prefix: run with JAVA or the dev shells' DATAMUNGE_JAVA_LIBRARY and MEP_JAVA_CLASSPATH.
build-java: prebuild-java
  {{ NIX_DEVELOP }} .#java --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungejava -B build/datamungejava -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#java --command bash -c 'cmake --build build/datamungejava -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#java --command bash -c 'rm -rf build/java/prefix && cmake --install build/datamungejava --prefix build/java/prefix'

# Installs the shared library libdatamunge_go (lib) and the module datamunge (share/go/datamunge)
# into build/go/prefix, then compiles the packages datamunge and datamunge/data from the committed
# module (as examples/go resolves them) into the Go build cache, so the first `go run` is quick.
build-go: prebuild-go
  {{ NIX_DEVELOP }} .#go --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungego -B build/datamungego -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#go --command bash -c 'cmake --build build/datamungego -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#go --command bash -c 'rm -rf build/go/prefix && cmake --install build/datamungego --prefix build/go/prefix'
  {{ NIX_DEVELOP }} .#go --command bash -c '{{ GO_ENV }} && go build datamunge datamunge/data'

build-d: prebuild-d
  # The native wrapper the generated module dlopens at startup (found via the dev shells' library path).
  {{ NIX_DEVELOP }} .#d --command bash -c 'mkdir -p build/d && ext="$(case "$(uname -s)" in Darwin) echo dylib;; *) echo so;; esac)" && c++ -std=c++23 -shared -fPIC -O1 $(pkg-config --cflags datamunge) -o "build/d/libdatamunge_wrap.$ext" {{ BINDINGS_DIR }}/datamunged/source/datamunged_wrap.cpp $(pkg-config --libs datamunge) -Wl,-rpath,$(pkg-config --variable=libdir datamunge)'
  bash -c 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -c "cd {{ BINDINGS_DIR }}/datamunged && dub build $compiler --build=release --force"; \
    else \
      cd {{ BINDINGS_DIR }}/datamunged && dub build $compiler --build=release --force; \
    fi'

# Installs Datamunge.pm and auto/Datamunge/libdatamunge_perl.so into build/perl/prefix/lib/perl5.
build-perl: prebuild-perl
  {{ NIX_DEVELOP }} .#perl --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungeperl -B build/datamungeperl -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#perl --command bash -c 'cmake --build build/datamungeperl -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#perl --command bash -c 'rm -rf build/perl/prefix && cmake --install build/datamungeperl --prefix build/perl/prefix >/dev/null'

# Installs datamunge.rb and the extension datamunge/datamunge.so into build/ruby/prefix/lib: run with
# RUBY (`ruby -I build/ruby/prefix/lib`) or the dev shells' RUBYLIB.
build-ruby: prebuild-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungeruby -B build/datamungeruby -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DRuby_EXECUTABLE="$(command -v ruby)" -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#ruby --command bash -c 'cmake --build build/datamungeruby -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#ruby --command bash -c 'rm -rf build/ruby/prefix && cmake --install build/datamungeruby --prefix build/ruby/prefix'

package-r: prebuild-r
  {{ NIX_DEVELOP }} .#r --command python3 scripts/build_r_source.py

build-r: prebuild-r
  {{ NIX_DEVELOP }} .#r --command bash -c 'mkdir -p build/r/library && R CMD INSTALL -l build/r/library .'
  # .Renviron lets R started here, and editors reading installed help (mep's K), find the package.
  if ! test -e .Renviron || head -1 .Renviron | grep -q '^# Written by just build-r'; then printf '# Written by just build-r\nR_LIBS_USER=%s/build/r/library\n' "$(pwd)" > .Renviron; fi

# Installs the package directory datamunge/ (pkgIndex.tcl, datamunge.tcl and the stubs extension
# datamunge.so) into build/tcl/prefix/lib: run with TCLSH or the dev shells' TCLLIBPATH.
build-tcl: prebuild-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungetcl -B build/datamungetcl -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DDATAMUNGE_TCLSH_EXECUTABLE="$(command -v tclsh)" -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#tcl --command bash -c 'cmake --build build/datamungetcl -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#tcl --command bash -c 'rm -rf build/tcl/prefix && cmake --install build/datamungetcl --prefix build/tcl/prefix'

build-lua: prebuild-lua
  {{ NIX_DEVELOP }} .#lua --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungelua -B build/datamungelua -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#lua --command bash -c 'cmake --build build/datamungelua -j{{ JOBS }} --verbose'
  {{ NIX_DEVELOP }} .#lua --command bash -c 'rm -rf build/lua/prefix && cmake --install build/datamungelua --prefix build/lua/prefix'

# Installs datamunge.asd, datamunge.lisp and datamunge.so into build/lisp/prefix, which the
# dev shells put on CL_SOURCE_REGISTRY: (asdf:load-system "datamunge"), package nickname dm.
build-lisp: prebuild-lisp
  {{ NIX_DEVELOP }} .#lisp --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungelisp -B build/datamungelisp -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#lisp --command bash -c 'cmake --build build/datamungelisp -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#lisp --command bash -c 'rm -rf build/lisp/prefix && cmake --install build/datamungelisp --prefix build/lisp/prefix'

test-lisp: build-lisp
  {{ NIX_DEVELOP }} .#lisp --command bash -c 'sbcl --noinform --non-interactive --load tests/lisp/test_datamunge.lisp'

# Installs the Datamunge package (Project.toml, src/Datamunge.jl, lib/datamunge.so) into
# build/julia/prefix/share/julia, which the dev shells put on JULIA_LOAD_PATH: `using Datamunge`.
build-julia: prebuild-julia
  {{ NIX_DEVELOP }} .#julia --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungejulia -B build/datamungejulia -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#julia --command bash -c 'cmake --build build/datamungejulia -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#julia --command bash -c 'rm -rf build/julia/prefix && cmake --install build/datamungejulia --prefix build/julia/prefix'

test-julia: build-julia
  {{ NIX_DEVELOP }} .#julia --command bash -c 'export JULIA_LOAD_PATH="$(pwd)/build/julia/prefix/share/julia:@:@v#.#:@stdlib" && julia --startup-file=no --threads=2 tests/julia/test_datamunge.jl'

# Installs libdatamungecobol and the copybooks into build/cobol/prefix (lib/ and
# share/cobol/copy/, which the dev shells put on COBCPY).
build-cobol: prebuild-cobol
  {{ NIX_DEVELOP }} .#cobol --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungecobol -B build/datamungecobol -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#cobol --command bash -c 'cmake --build build/datamungecobol -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#cobol --command bash -c 'rm -rf build/cobol/prefix && cmake --install build/datamungecobol --prefix build/cobol/prefix'

test-cobol: build-cobol
  {{ NIX_DEVELOP }} .#cobol --command bash -c 'mkdir -p build/cobol/bin && {{ COBC }} -o build/cobol/bin/test_datamunge tests/cobol/test_datamunge.cob && build/cobol/bin/test_datamunge'

# Installs libdatamunge_rust into build/rust/prefix/lib and the crate into build/rust/prefix/share/cargo/datamunge.
build-rust: prebuild-rust
  {{ NIX_DEVELOP }} .#rust --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungerust -B build/datamungerust -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#rust --command bash -c 'cmake --build build/datamungerust -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#rust --command bash -c 'rm -rf build/rust/prefix && cmake --install build/datamungerust --prefix build/rust/prefix >/dev/null'

# Installs libdatamunge_haskell and the GHC package datamunge (datamunge.cabal; lib/haskell, registered in
# lib/haskell/package.db) into build/haskell/prefix.
build-haskell: prebuild-haskell
  {{ NIX_DEVELOP }} .#haskell --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungehaskell -B build/datamungehaskell -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$(pwd)/build/haskell/prefix" -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#haskell --command bash -c 'cmake --build build/datamungehaskell -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#haskell --command bash -c 'cmake --install build/datamungehaskell >/dev/null'

# Installs libdatamunge_prolog (lib) and the module (prolog/datamunge.pl) into build/prolog/prefix.
build-prolog: prebuild-prolog
  {{ NIX_DEVELOP }} .#prolog --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungeprolog -B build/datamungeprolog -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#prolog --command bash -c 'cmake --build build/datamungeprolog -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#prolog --command bash -c 'rm -rf build/prolog/prefix && cmake --install build/datamungeprolog --prefix build/prolog/prefix >/dev/null'

# Installs libdatamunge_objc, the classes libDM (clang, ARC, GNUstep 2.0 runtime) and include/DM/DM.h into
# build/objc/prefix.
build-objc: prebuild-objc
  {{ NIX_DEVELOP }} .#objc --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungeobjc -B build/datamungeobjc -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#objc --command bash -c 'cmake --build build/datamungeobjc -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#objc --command bash -c 'rm -rf build/objc/prefix && cmake --install build/datamungeobjc --prefix build/objc/prefix >/dev/null'

# Installs libdatamunge_ada and the compiled package (lib/ada/datamunge: .ads, .adb, .gpr, .ali, .o; the
# package takes about a minute to compile) into build/ada/prefix.
build-ada: prebuild-ada
  {{ NIX_DEVELOP }} .#ada --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungeada -B build/datamungeada -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#ada --command bash -c 'cmake --build build/datamungeada -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#ada --command bash -c 'rm -rf build/ada/prefix && cmake --install build/datamungeada --prefix build/ada/prefix >/dev/null'

# Installs libdatamunge_dart and the package (share/dart/datamunge) into build/dart/prefix.
build-dart: prebuild-dart
  {{ NIX_DEVELOP }} .#dart --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungedart -B build/datamungedart -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#dart --command bash -c 'cmake --build build/datamungedart -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#dart --command bash -c 'rm -rf build/dart/prefix && cmake --install build/datamungedart --prefix build/dart/prefix >/dev/null'

# Installs libdatamunge_pascal and the compiled unit (lib/fpc/datamunge: datamunge.ppu, .o, .pas) into
# build/pascal/prefix.
build-pascal: prebuild-pascal
  {{ NIX_DEVELOP }} .#pascal --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungepascal -B build/datamungepascal -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#pascal --command bash -c 'cmake --build build/datamungepascal -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#pascal --command bash -c 'rm -rf build/pascal/prefix && cmake --install build/datamungepascal --prefix build/pascal/prefix >/dev/null'

# Installs libdatamunge_swift, the Swift module (libDatamunge.so, lib/swift/Datamunge.swiftmodule) and
# the C module share/swift/CDatamunge into build/swift/prefix.
build-swift: prebuild-swift
  {{ NIX_DEVELOP }} .#swift --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungeswift -B build/datamungeswift -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#swift --command bash -c 'cmake --build build/datamungeswift -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#swift --command bash -c 'rm -rf build/swift/prefix && cmake --install build/datamungeswift --prefix build/swift/prefix >/dev/null'

# Installs datamunge.scm, datamunge/ and libdatamunge_guile.so into
# build/guile/prefix/share/guile/site/<version>, which GUILE (run-guile, test-guile, repl-guile) and
# the dev shells' GUILE_LOAD_PATH put on the load path.
build-guile: prebuild-guile
  {{ NIX_DEVELOP }} .#guile --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungeguile -B build/datamungeguile -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#guile --command bash -c 'cmake --build build/datamungeguile -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#guile --command bash -c 'rm -rf build/guile/prefix && cmake --install build/datamungeguile --prefix build/guile/prefix >/dev/null'

# Installs __datamunge__.oct and the package +datamunge into build/octave/prefix/share/octave/site/m,
# which OCTAVE (run-octave, test-octave, repl-octave) and the dev shells' OCTAVE_PATH put on the path.
build-octave: prebuild-octave
  {{ NIX_DEVELOP }} .#octave --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungeoctave -B build/datamungeoctave -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#octave --command bash -c 'cmake --build build/datamungeoctave -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#octave --command bash -c 'rm -rf build/octave/prefix && cmake --install build/datamungeoctave --prefix build/octave/prefix >/dev/null'

# Installs the findlib package datamunge/ (META, datamunge.cma/.cmxa/.cmi/.mli, dlldatamunge_stubs.so and
# libdatamunge_stubs.a) into build/ocaml/prefix/lib: use with OCAML_ENV or the dev shells' OCAMLPATH.
build-ocaml: prebuild-ocaml
  {{ NIX_DEVELOP }} .#ocaml --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungeocaml -B build/datamungeocaml -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#ocaml --command bash -c 'cmake --build build/datamungeocaml -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#ocaml --command bash -c 'rm -rf build/ocaml/prefix && cmake --install build/datamungeocaml --prefix build/ocaml/prefix >/dev/null'



# Installs libdatamunge_gnu_apl.so into build/apl/prefix/lib and the script datamunge.apl into
# build/apl/prefix/share/apl: use with APL or the dev shells' DATAMUNGE_APL_LIBRARY.
build-apl: prebuild-apl
  {{ NIX_DEVELOP }} .#apl --command bash -c 'datamunge_prefix="$(pkg-config --variable=prefix datamunge)" && cmake -S src/datamungeapl -B build/datamungeapl -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -U datamunge_DIR -DROSETTA_GNU_APL_SOURCE_DIR="$ROSETTA_GNU_APL_SOURCE_DIR" -DCMAKE_PREFIX_PATH="$datamunge_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"'
  {{ NIX_DEVELOP }} .#apl --command bash -c 'cmake --build build/datamungeapl -j{{ JOBS }}'
  {{ NIX_DEVELOP }} .#apl --command bash -c 'rm -rf build/apl/prefix && cmake --install build/datamungeapl --prefix build/apl/prefix >/dev/null'


# }}} build commands

# {{{ repl commands

repl-javascript: prebuild-javascript build-javascript
  {{ NIX_DEVELOP }} .#javascript --command bash -c 'node'

repl-python: build-python
  {{ NIX_DEVELOP }} .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python

repl-r: build-r
  {{ NIX_DEVELOP }} .#r --command bash -c 'R_LIBS="$(pwd)/build/r/library" R'

# php -a with the binding loaded (src/datamungephp/repl.php: the namespace Datamunge imported).
repl-php: build-php
  {{ NIX_DEVELOP }} .#php --command bash -c 'php -d auto_prepend_file="$(pwd)/src/datamungephp/repl.php" -a'

repl-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -c 'export PERL5LIB="$(pwd)/build/perl/prefix/lib/perl5${PERL5LIB:+:$PERL5LIB}" && perl -MDatamunge=:all -de 1'

# dotnet-repl with Datamunge.dll referenced (`#r`) and `using Datamunge;` (mep's C# console: `#:project` in a file).
repl-csharp: build-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'printf "#r \"%s\"\nusing Datamunge;\n" "$(pwd)/build/csharp/prefix/share/dotnet/Datamunge/bin/Datamunge.dll" > build/csharp/repl.csx && dotnet-repl --run build/csharp/repl.csx'

# jshell with datamunge.jar on the class path, DATAMUNGE_JAVA_LIBRARY set and `import datamunge.*;`
# (src/datamungejava/repl.jsh): new DataFrame(), Datamunge.seql(0, 1, 5), ...
repl-java: build-java
  {{ NIX_DEVELOP }} .#java --command bash -c 'export DATAMUNGE_JAVA_LIBRARY="$(pwd)/build/java/prefix/lib/libdatamunge_java.so" && jshell --class-path "$(pwd)/build/java/prefix/share/java/datamunge.jar" DEFAULT src/datamungejava/repl.jsh'

# C++ REPL on clang-repl, with the std headers and <datamunge/datamunge.hpp>
# already included. Optional EXAMPLE preloads examples/cpp/source/<topic>/<EXAMPLE>.cpp
# and calls its main(), e.g. `just repl-cpp dataframe_ex`.
repl-cpp EXAMPLE="":
  {{ NIX_DEVELOP }} .#cpp --command bash -lc 'exec scripts/cpp_repl.sh {{ EXAMPLE }}'

# Same, but on cling instead of clang-repl.
repl-cling EXAMPLE="":
  {{ NIX_DEVELOP }} .#cpp --command bash -lc 'exec scripts/cpp_repl.sh --cling {{ EXAMPLE }}'

# tclsh with the package loaded (src/datamungetcl/repl.tcl): datamunge::DataFrame new, datamunge::seql 0 1 5, ...
repl-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -c '{{ TCLSH }} src/datamungetcl/repl.tcl'

repl-lua: build-lua
  {{ NIX_DEVELOP }} .#lua --command bash -c 'lua'

repl-lisp: build-lisp
  {{ NIX_DEVELOP }} .#lisp --command bash -c 'sbcl --eval "(require \"asdf\")" --eval "(asdf:load-system \"datamunge\")"'

repl-julia: build-julia
  {{ NIX_DEVELOP }} .#julia --command bash -c 'export JULIA_LOAD_PATH="$(pwd)/build/julia/prefix/share/julia:@:@v#.#:@stdlib" && julia --threads=auto -i -e "using Datamunge"'

# irb with the binding loaded: Datamunge::DataFrame, Datamunge.seql(0, 1, 5), ...
repl-ruby: build-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -c 'irb -I "$(pwd)/build/ruby/prefix/lib" -r datamunge'

# utop with the binding loaded (src/datamungeocaml/repl.ml: topfind, `#require "datamunge"`, `open
# Datamunge`): DataFrame.create (), seql 0. 1. 5, ... utop needs a terminal: with piped input the plain
# ocaml toplevel reads it instead.
repl-ocaml: build-ocaml
  {{ NIX_DEVELOP }} .#ocaml --command bash -c '{{ OCAML_ENV }} && if [ -t 0 ]; then utop -init src/datamungeocaml/repl.ml; else ocaml -noprompt -init src/datamungeocaml/repl.ml; fi'

# GNU APL with the binding loaded (src/datamungeapl/repl.apl prints a banner): dm∆seql 0 1 5,
# df←dm∆DataFrame∆new ⍬, ⎕CR 'dm∆seql' for the documentation, )OFF to leave.
repl-apl: build-apl
  {{ NIX_DEVELOP }} .#apl --command bash -c 'export DATAMUNGE_APL_LIBRARY="$(pwd)/build/apl/prefix/lib/libdatamunge_gnu_apl.so" && apl --noSV --silent --noCONT -f "$(pwd)/build/apl/prefix/share/apl/datamunge.apl" -f src/datamungeapl/repl.apl'

# guile's REPL with (datamunge) loaded: try `(make <data-frame>)` or `,describe seql`.
repl-guile: build-guile
  {{ NIX_DEVELOP }} .#guile --command bash -c '{{ GUILE }} -l src/datamungeguile/repl.scm'

# evcxr with build/rust/evcxr as its configuration directory: init.evcxr adds the crate as a dependency
# and imports it (the first evaluation builds it).
repl-rust: build-rust
  {{ NIX_DEVELOP }} .#rust --command bash -c '{{ RUST_ENV }} && mkdir -p build/rust/evcxr && printf ":dep datamunge = { path = \"%s\" }\nuse datamunge::*;\n" "$(pwd)/src/datamungerust/datamunge" > build/rust/evcxr/init.evcxr && EVCXR_CONFIG_DIR="$(pwd)/build/rust/evcxr" evcxr'

# octave-cli with the binding on the path: try `df = datamunge.DataFrame ()` or `help datamunge.seql`.
repl-octave: build-octave
  {{ NIX_DEVELOP }} .#octave --command bash -c '{{ OCTAVE }}" --persist'

repl-d:
  bash -c 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -c "cd examples/d && dub run $compiler --build=release"; \
    else \
      cd examples/d && dub run $compiler --build=release; \
    fi'

# }}} repl commands

# {{{ test commands

test-python-build: prebuild-python
  rm -rf build/venv/pydatamunge-build
  {{ NIX_DEVELOP }} .#python --command bash -c 'python -m venv --system-site-packages build/venv/pydatamunge-build'
  {{ NIX_DEVELOP }} .#python --command bash -c 'build/venv/pydatamunge-build/bin/python setup.py sdist bdist_wheel'

# Test installed packages, with no source-tree override or older installed binding.
test-python: build-python
  {{ NIX_DEVELOP }} .#python --command bash -c 'env -u PYTHONPATH PYTHONNOUSERSITE=1 build/venv/pydatamunge-run/bin/python -m pytest --import-mode=importlib -q tests/python bindings_tests/python'

test-r: build-r
  {{ NIX_DEVELOP }} .#r --command Rscript -e 'library(datamunger, lib.loc="build/r/library"); stopifnot(startsWith(find.package("datamunger"), normalizePath("build/r/library"))); testthat::test_dir("tests/testthat", stop_on_failure=TRUE); testthat::test_dir("bindings_tests/r", stop_on_failure=TRUE)'


test-csharp: build-csharp
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'dotnet run tests/csharp/test_datamunge.cs'

test-vb: build-vb
  {{ NIX_DEVELOP }} .#csharp --command bash -c 'dotnet run --project tests/vb/Test.vbproj'

# Compiles tests/fortran/test_datamunge.f90 against build/fortran/prefix and runs it.
test-fortran: build-fortran
  {{ NIX_DEVELOP }} .#fortran --command bash -c '{{ FORTRAN_FC }} tests/fortran/test_datamunge.f90 {{ FORTRAN_LIBS }} -o build/fortran/bin/test_datamunge && build/fortran/bin/test_datamunge'

# tests/java/TestDatamunge.java (a single-file launch) against build/java/prefix; it reads the
# committed sources src/datamungejava/java for the documentation checks.
test-java: build-java
  {{ NIX_DEVELOP }} .#java --command bash -c '{{ JAVA }} tests/java/TestDatamunge.java src/datamungejava/java'

test-rust: build-rust
  {{ NIX_DEVELOP }} .#rust --command bash -c '{{ RUST_ENV }} && cargo test --offline --manifest-path tests/rust/Cargo.toml'

test-haskell: build-haskell
  {{ NIX_DEVELOP }} .#haskell --command bash -c '{{ GHC }} -outputdir build/haskell/obj/test_datamunge tests/haskell/test_datamunge.hs -o build/haskell/bin/test_datamunge && build/haskell/bin/test_datamunge'

# GHCi with the GHC package datamunge.
repl-haskell: build-haskell
  {{ NIX_DEVELOP }} .#haskell --command bash -c 'ghci -package-db "$(pwd)/build/haskell/prefix/lib/haskell/package.db" -package datamunge -ghci-script <(echo "import Datamunge")'

test-prolog: build-prolog
  {{ NIX_DEVELOP }} .#prolog --command bash -c '{{ SWIPL }} tests/prolog/test_datamunge.pl'

# SWI-Prolog toplevel with the module loaded.
repl-prolog: build-prolog
  {{ NIX_DEVELOP }} .#prolog --command bash -c '{{ SWIPL }} -g "use_module(library(datamunge))"'

test-objc: build-objc
  {{ NIX_DEVELOP }} .#objc --command bash -c '{{ OBJCC }} tests/objc/test_datamunge.m -o build/objc/bin/test_datamunge && build/objc/bin/test_datamunge'

test-ada: build-ada
  {{ NIX_DEVELOP }} .#ada --command bash -c '{{ GNATMAKE }} tests/ada test_datamunge.adb && build/ada/bin/test_datamunge'

test-dart: build-dart
  {{ NIX_DEVELOP }} .#dart --command bash -c '{{ DART_RUN }} tests/dart tests/dart/bin/test_datamunge.dart'

test-pascal: build-pascal
  {{ NIX_DEVELOP }} .#pascal --command bash -c '{{ FPC }} tests/pascal/test_datamunge.pas && build/pascal/bin/test_datamunge'

test-swift: build-swift
  {{ NIX_DEVELOP }} .#swift --command bash -c '{{ SWIFTC }} tests/swift/test_datamunge.swift -o build/swift/bin/test_datamunge && build/swift/bin/test_datamunge'

test-php: build-php
  {{ NIX_DEVELOP }} .#php --command bash -c 'php tests/php/test_datamunge.php'

test-lua: build-lua
  {{ NIX_DEVELOP }} .#lua --command bash -c 'lua tests/lua/test_datamunge.lua'

test-perl: build-perl
  {{ NIX_DEVELOP }} .#perl --command bash -c 'export PERL5LIB="$(pwd)/build/perl/prefix/lib/perl5${PERL5LIB:+:$PERL5LIB}" && prove tests/perl'

# tests/tcl/test_datamunge.tcl against build/tcl/prefix; it reads the committed
# src/datamungetcl/pkg/datamunge/datamunge.tcl for the documentation checks.
test-tcl: build-tcl
  {{ NIX_DEVELOP }} .#tcl --command bash -c '{{ TCLSH }} tests/tcl/test_datamunge.tcl src/datamungetcl/pkg/datamunge/datamunge.tcl'

# tests/ruby/test_datamunge.rb against build/ruby/prefix; it reads the committed
# src/datamungeruby/lib/datamunge.rb for the YARD documentation checks.
test-ruby: build-ruby
  {{ NIX_DEVELOP }} .#ruby --command bash -c '{{ RUBY }} tests/ruby/test_datamunge.rb src/datamungeruby/lib/datamunge.rb'

# tests/guile/test_datamunge.scm against build/guile/prefix; it reads the committed
# src/datamungeguile/scheme/datamunge.scm for the docstring checks.
test-guile: build-guile
  {{ NIX_DEVELOP }} .#guile --command bash -c '{{ GUILE }} -s tests/guile/test_datamunge.scm src/datamungeguile/scheme/datamunge.scm'

# tests/octave/test_datamunge.m against build/octave/prefix (including the generated help text).
test-octave: build-octave
  {{ NIX_DEVELOP }} .#octave --command bash -c '{{ OCTAVE }} source tests/octave/test_datamunge.m"'

test-d: prebuild-d
  bash -c 'set -euo pipefail; \
    compiler=""; \
    if command -v ldc2 >/dev/null 2>&1; then compiler="--compiler=ldc2"; elif command -v dmd >/dev/null 2>&1; then compiler="--compiler=dmd"; fi; \
    if command -v nix >/dev/null 2>&1 && {{ NIX_DEVELOP }} .#d --command true >/dev/null 2>&1; then \
      {{ NIX_DEVELOP }} .#d --command bash -c "dub test --root tests/d $compiler --build=release"; \
    else \
      dub test --root tests/d $compiler --build=release; \
    fi'


# tests/go (go test in the module tests/go) against build/go/prefix; it reads the committed module
# src/datamungego/go/datamunge for the documentation checks.
# tests/apl/test_datamunge.apl against build/apl/prefix; it reads the committed
# src/datamungeapl/apl/datamunge.apl (DATAMUNGE_APL_DOCS) for the documentation checks.
test-apl: build-apl
  {{ NIX_DEVELOP }} .#apl --command bash -c 'set -o pipefail; export DATAMUNGE_APL_DOCS="$(pwd)/src/datamungeapl/apl/datamunge.apl"; {{ APL_RUN }}; apl_run tests/apl/test_datamunge.apl && grep -q "^DATAMUNGE APL PASS" build/apl/log/test_datamunge.log'

test-go: build-go
  {{ NIX_DEVELOP }} .#go --command bash -c '{{ GO_ENV }} && cd tests/go && go test -count=1 -v .'


test-javascript: build-javascript
    @echo "Running Javascript Tests"
    {{ NIX_DEVELOP }} .#javascript --command bash -c "npm run test"

# tests/ocaml/test_datamunge.ml, compiled natively against build/ocaml/prefix; it reads the committed
# src/datamungeocaml/ocaml/datamunge/datamunge.mli for the documentation checks.
test-ocaml: build-ocaml
  {{ NIX_DEVELOP }} .#ocaml --command bash -c '{{ OCAML_ENV }} && {{ OCAML_BUILD }}; ocaml_build tests/ocaml/test_datamunge.ml build/ocaml/bin/test_datamunge && ./build/ocaml/bin/test_datamunge src/datamungeocaml/ocaml/datamunge/datamunge.mli'


test-cpp:
    @echo "Running Tests"
    @bash -c 'set -euo pipefail; \
      gtest_prefix=""; \
      for p in /nix/store/*-gtest-*-dev; do \
        if [ -f "$p/lib/cmake/GTest/GTestConfig.cmake" ]; then gtest_prefix="$p"; break; fi; \
      done; \
      rm -rf build/debug/tests; \
      cmake -S tests -B build/debug/tests -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_MAKE_PROGRAM="$(command -v make)" ${gtest_prefix:+-DCMAKE_PREFIX_PATH="$gtest_prefix"}; \
      cmake --build build/debug/tests -j{{ JOBS }} --verbose; \
      find ./build/ -name "compile_commands.json" -exec cat {} + | jq -s add > compile_commands.json; \
      ctest --test-dir build/debug/tests --output-on-failure'

# Incrementally rebuild the C++ tests and run only the nn (tensor + autograd) suites.
test-nn:
  {{ NIX_DEVELOP }} .#cpp-tools --command bash -c "cmake --build build/debug/tests -j{{ NN_JOBS }} && ./build/debug/tests/run_tests --gtest_filter='Nn*'"


# }}} test commands

# {{{ nn examples (datamunge::nn ports of ~/projects/tflow_examples)

# The standalone C++ tree is retired; resolve names to the verified literate reports.
run-nn NAME *ARGS:
  python3 scripts/run_nn_examples.py --force {{ quote(NAME) }} --example-args {{ ARGS }}

run-all-nn *ARGS:
  python3 scripts/run_nn_examples.py {{ ARGS }}

# Optional SWIG-backed replicas of the migrated CNN / autoencoder sections.
prebuild-nn-python: prebuild-python
  {{ NIX_DEVELOP }} .#python --command python scripts/prebuild_nn_api.py python
  {{ NIX_DEVELOP }} .#python --command python scripts/prebuild_nn_examples.py

build-nn-python: prebuild-nn-python
  # Recreate the venv when its interpreter is gone (a nix garbage collection removes the shell's python).
  {{ NIX_DEVELOP }} .#python --command bash -c 'build/venv/pydatamunge-run/bin/python -c "" 2>/dev/null || python -m venv --clear --system-site-packages build/venv/pydatamunge-run'
  # .venv lets editors (mep's Python hover/completion) find the interpreter outside the dev shell.
  test -e .venv || ln -s build/venv/pydatamunge-run .venv
  {{ NIX_DEVELOP }} .#python --command bash -c 'DATAMUNGE_BUILD_NN_EXAMPLES=1 CMAKE_BUILD_PARALLEL_LEVEL={{ NN_JOBS }} build/venv/pydatamunge-run/bin/python -m pip install --force-reinstall . --no-deps --no-build-isolation'
  touch build/nn-python/enabled

test-nn-python: build-nn-python
  {{ NIX_DEVELOP }} .#python --command build/venv/pydatamunge-run/bin/python -m pytest --import-mode=importlib -q tests/python/test_nn_language_examples.py tests/python/test_nn_functional_api.py

run-nn-python NAME *ARGS: build-nn-python
  {{ NIX_DEVELOP }} .#python --command build/venv/pydatamunge-run/bin/python scripts/run_nn_python_examples.py --force {{ NAME }} --example-args {{ ARGS }}

run-nn-python-full NAME *ARGS: build-nn-python
  {{ NIX_DEVELOP }} .#python --command build/venv/pydatamunge-run/bin/python scripts/run_nn_python_examples.py --full --force {{ NAME }} --example-args {{ ARGS }}

run-all-nn-python *ARGS: build-nn-python
  {{ NIX_DEVELOP }} .#python --command build/venv/pydatamunge-run/bin/python scripts/run_nn_python_examples.py {{ ARGS }}

prebuild-nn-r:
  {{ NIX_DEVELOP }} .#r --command Rscript --vanilla scripts/prebuild_nn_r.R

build-nn-r: prebuild-nn-r
  mkdir -p build/r/library
  {{ NIX_DEVELOP }} .#r --command env DATAMUNGE_USE_SYSTEM_LIB=1 MAKEFLAGS=-j{{ NN_JOBS }} R CMD INSTALL -l build/r/library build/nn-r-native/source/datamungerNN

package-nn-r: build-nn-r
  {{ NIX_DEVELOP }} .#r --command bash -c 'mkdir -p dist && cd dist && R CMD build --no-manual ../build/nn-r-native/source/datamungerNN'

test-nn-r: build-nn-r
  {{ NIX_DEVELOP }} .#r --command Rscript -e 'library(datamungerNN, lib.loc="build/r/library"); testthat::test_dir("bindings_tests/r-nn", stop_on_failure=TRUE)'

run-nn-r NAME *ARGS: build-nn-r
  {{ NIX_DEVELOP }} .#r --command env R_LIBS_USER=build/r/library DATAMUNGE_EXAMPLES_FULL=0 Rscript --vanilla -e 'args <- commandArgs(TRUE); quit(status=datamungerNN::nn_example_run(args[[1L]], args[-1L]))' {{ NAME }} {{ ARGS }}

run-nn-r-full NAME *ARGS: build-nn-r
  {{ NIX_DEVELOP }} .#r --command env R_LIBS_USER=build/r/library DATAMUNGE_EXAMPLES_FULL=1 Rscript --vanilla -e 'args <- commandArgs(TRUE); quit(status=datamungerNN::nn_example_run(args[[1L]], args[-1L]))' {{ NAME }} {{ ARGS }}

run-all-nn-r *ARGS: build-nn-r
  {{ NIX_DEVELOP }} .#r --command env R_LIBS_USER=build/r/library Rscript --vanilla -e 'for (name in datamungerNN::nn_example_names()) { cat("\n==", name, "==\n"); status <- datamungerNN::nn_example_run(name, commandArgs(TRUE)); if (status != 0L) quit(status=status) }' {{ ARGS }}

# Genuine language implementations; these commands never build C++ experiment adapters.
build-nn-api-python: prebuild-python
  # Recreate the venv when its interpreter is gone (a nix garbage collection removes the shell's python).
  {{ NIX_DEVELOP }} .#python --command bash -c 'build/venv/pydatamunge-run/bin/python -c "" 2>/dev/null || python -m venv --clear --system-site-packages build/venv/pydatamunge-run'
  # .venv lets editors (mep's Python hover/completion) find the interpreter outside the dev shell.
  test -e .venv || ln -s build/venv/pydatamunge-run .venv
  {{ NIX_DEVELOP }} .#python --command bash -c 'DATAMUNGE_USE_SYSTEM_LIB=1 DATAMUNGE_BUILD_NN_EXAMPLES=0 CMAKE_BUILD_PARALLEL_LEVEL={{ NN_JOBS }} build/venv/pydatamunge-run/bin/python -m pip install . --no-deps --no-build-isolation'

# Native workflows and reusable SWIG components; excludes the optional C++ experiment bridge.
test-nn-api-python: build-nn-api-python
  {{ NIX_DEVELOP }} .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python -m pytest --import-mode=importlib -q tests/python/test_nn_example_regressions.py tests/python/test_nn_dataset_assets.py tests/python/test_nn_cifar_workflows.py tests/python/test_nn_memory_workflows.py tests/python/test_nn_components.py tests/python/test_nn_example_workflow_ownership.py tests/python/test_nn_functional_api.py tests/python/test_nn_migration_inventory.py

python-example NAME *ARGS: build-nn-api-python
  {{ NIX_DEVELOP }} .#python --command build/venv/pydatamunge-run/bin/python -I scripts/python_example.py {{ NAME }} {{ ARGS }}

r-example NAME *ARGS: build-nn-r
  {{ NIX_DEVELOP }} .#r --command env R_LIBS_USER=build/r/library Rscript --vanilla -e 'args <- commandArgs(TRUE); quit(status=datamungerNN::nn_example_run(args[[1L]], args[-1L]))' {{ NAME }} {{ ARGS }}

nn-language-status:
  {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/nn_migration_status.py --refresh

# Original MNIST IDX files plus balanced, disjoint pools used by autoencoder examples.
nn-mnist-assets:
  {{ NIX_DEVELOP }} .#cpp --command datamunge-prepare-data mnist-assets

nn-mnist-assets-r: build-nn-r
  {{ NIX_DEVELOP }} .#r --command env R_LIBS_USER=build/r/library Rscript --vanilla -e 'root <- Sys.getenv("DATAMUNGE_DATA_DIR", "build/datasets"); datamungerNN::nn_components$dataset_assets$EnsureMnist(file.path(root,"mnist")); datamungerNN::nn_components$dataset_assets$EnsureSaliency(file.path(root,"saliency"))'

# Exhaustive binary circle/square factors and fixed compositional holdouts.
nn-beta-vae-assets:
  {{ NIX_DEVELOP }} .#cpp --command datamunge-prepare-data beta-vae-assets

# Prepare the exhaustive local compositional sentence grammar in the shared cache.
nn-sequence-vae-assets:
  {{ NIX_DEVELOP }} .#cpp --command datamunge-prepare-data sequence-vae-assets

nn-memory-assets:
  {{ NIX_DEVELOP }} .#cpp --command datamunge-prepare-data memory

nn-memory-assets-r: build-nn-r
  {{ NIX_DEVELOP }} .#r --command env R_LIBS_USER=build/r/library Rscript --vanilla -e 'root <- Sys.getenv("DATAMUNGE_DATA_DIR", "build/datasets"); datamungerNN::nn_components$memory_components$PrepareData(root, FALSE); cat("Prepared local synthetic memory tasks in", root, "\n")'

# Where each example keeps its workflow (main / local function / pasted or header harness),
# which shared headers it uses and whether they are SWIG-bound. Writes build/nn-example-audit.json
# and docs/nn-example-audit.md; --check fails when a shared header owns a training loop.
nn-audit *ARGS:
  {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/nn_example_audit.py --markdown docs/nn-example-audit.md {{ ARGS }}

# Run C++/Python/R for each NAME (epochs=5) and record passing evidence in the ledger.
validate-nn-ports *ARGS:
  {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/validate_nn_ports.py {{ ARGS }}

# Exact persisted-array comparison across the three language output directories.
nn-compare *ARGS:
  {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/compare_nn_outputs.py {{ ARGS }}

# }}} nn examples

# {{{ utilities

rename NEW:
  ./rename_datamunge {{ NEW }}

jq:
    {{ NIX_DEVELOP }} . --command bash -c "find ./build -name 'compile_commands.json' -exec cat {} + | jq -s add > compile_commands.json"


# Run the playground document, print its captured results, and export HTML/TeX.
# Optional arguments select export formats; the C++ demo remains TARGET=playground run-cpp.
[positional-arguments]
playground *FORMATS:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "$#" -eq 0 ]; then set -- html tex; fi
    output_dir="$PWD/build/playground"
    document="docs/mepml/playground.mepml"
    mkdir -p "$output_dir"
    pc="{{ MEPML_PKG_CONFIG_PATH }}"
    {{ NIX_DEVELOP }} .#cpp --command env -u LD_LIBRARY_PATH ${pc:+PKG_CONFIG_PATH="$pc"} ${pc:+DATAMUNGE_CAD_PKG_CONFIG_PATH="$pc"} \
      mep-mepml build --run --timeout "${DATAMUNGE_MEPML_TIMEOUT:-7200}" --report "$output_dir/run.json" --no-post \
      -d "$output_dir" "$document" "$@"
    mkdir -p "$output_dir/figures"
    if [ -d "$(dirname "$document")/figures" ]; then
      cp -R "$(dirname "$document")/figures/." "$output_dir/figures/"
    fi
    awk '/^\/\/ result_begin/ {printing=1; next} /^\/\/ result_end/ {printing=0; print ""; next} printing {sub(/^\/\/ ?/, ""); print}' "$document"
    printf 'Build output: %s\n' "$output_dir"

# Frozen-checkpoint preview; the report's default window is 40..60 seconds.
[positional-arguments]
playground-preview *FORMATS:
    #!/usr/bin/env bash
    set -euo pipefail
    export DATAMUNGE_STREAM_REPLAY=1
    just playground "$@"

# Twenty-second CUDA overview preview; sensor inference remains the CPU reference.
[positional-arguments]
playground-gpu *FORMATS:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "${DATAMUNGE_MEPML_PKG_CONFIG_PATH:-}" ]; then
      core="$(nix build --no-link --print-out-paths .#datamunge-cuda)"
      export DATAMUNGE_MEPML_PKG_CONFIG_PATH="$core/lib/pkgconfig"
    fi
    export DATAMUNGE_RENDER_BACKEND=cuda
    export DATAMUNGE_VIDEO_START="${DATAMUNGE_VIDEO_START:-40}"
    export DATAMUNGE_VIDEO_SECONDS="${DATAMUNGE_VIDEO_SECONDS:-20}"
    just playground-preview "$@"

flamechart:
    @echo "Running Performance Tests"
    @perf record -F 99 -g ./build/examples/${TARGET}
    @perf script > out.perf
    @if [ ! -d "Flamegraph" ]; then \
    	git clone https://github.com/brendangregg/Flamegraph.git; \
    fi
    @./Flamegraph/stackcollapse-perf.pl out.perf > out.folded
    @./Flamegraph/flamegraph.pl out.folded > flamegraph.svg


benchmark:
    @echo "Building benchmarks"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/benchmarks --config Release -j{{ JOBS }}"
    mkdir -p benchmarks/results
    @echo "Running {{ BENCH_TARGET }}"
    python3 scripts/benchmark_backend.py run --no-build --suite {{ BENCH_TARGET }}
    @echo "Results saved to benchmarks/results/"

# Replay bounded rendering/physics/inference cases using the realistic playground's saved run.
# Requires one completed realistic playground; does not train or modify the report/artifacts.
bench-fruit-render *ARGS:
    {{ NIX_DEVELOP }} .#cpp --command python3 scripts/benchmark_fruit_render.py {{ ARGS }}

# Four CPU wrist poses with exact buffers/crops/boxes; no video or model training.
[positional-arguments]
bench-fruit-wrist *ARGS:
    {{ NIX_DEVELOP }} .#cpp --command python3 scripts/benchmark_fruit_wrist.py "$@"

# Four matched CPU/CUDA poses, at most five repeats; no video encoding or training.
[positional-arguments]
bench-fruit-render-gpu *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    pc="${DATAMUNGE_MEPML_PKG_CONFIG_PATH:-}"
    if [ -z "$pc" ]; then
      core="$(nix build --no-link --print-out-paths .#datamunge-cuda)"
      pc="$core/lib/pkgconfig"
    fi
    {{ NIX_DEVELOP }} .#cpp-tools --command env PKG_CONFIG_PATH="$pc" python3 scripts/benchmark_fruit_render_gpu.py "$@"

# SqueezeNet CIFAR-10 training throughput (epochs per minute); see benchmarks/nn/squeezenet_train.cpp.
# Extra arguments go to the driver, e.g. `just bench-squeezenet --threads 16 --seconds 60`.
bench-squeezenet *ARGS:
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/benchmarks --config Release --target squeezenet_train -j{{ JOBS }}"
    {{ NIX_DEVELOP }} .#cpp --command ./build/benchmarks/squeezenet_train {{ ARGS }}

# Per-op micro-benchmarks at SqueezeNet's exact layer shapes, 1 and 16 threads (see benchmarks/SQUEEZENET.md).
# Extra arguments go to Google Benchmark, e.g. `just bench-squeezenet-ops --benchmark_filter=Conv2dForward`.
bench-squeezenet-ops *ARGS:
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/benchmarks --config Release --target squeezenet_ops_bench -j{{ JOBS }}"
    {{ NIX_DEVELOP }} .#cpp --command ./build/benchmarks/squeezenet_ops_bench {{ ARGS }}

# In-house BLAS (every instruction set this CPU runs) against OpenBLAS (nixpkgs Haswell
# kernels and an AVX-512 build), BLIS and the netlib reference BLAS from the blas-bench
# shell: GEMM at square and SqueezeNet shapes, GEMV, DOT, AXPY, SCAL, float and double,
# 1/8/16 threads. Writes benchmarks/results/blas/cpu.json and cpu.md (see benchmarks/blas/).
# Extra arguments go to the driver, e.g. `just bench-blas --quick --threads 1,8`.
bench-blas *ARGS:
    {{ NIX_DEVELOP }} .#blas-bench --command bash -c "cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
    {{ NIX_DEVELOP }} .#blas-bench --command bash -c "cmake --build build/benchmarks --config Release --target blas_compare -j{{ JOBS }}"
    mkdir -p benchmarks/results/blas
    {{ NIX_DEVELOP }} .#blas-bench --command ./build/benchmarks/blas_compare --json benchmarks/results/blas/cpu.json {{ ARGS }}
    python3 benchmarks/blas/summarize.py benchmarks/results/blas/cpu.json --out benchmarks/results/blas/cpu.md

# Built-in decompositions (Cholesky, LU solve, slogdet, triangular solve, QR, eigh, SVD) against
# LAPACK in one -DDATAMUNGE_USE_LAPACK=ON build. The library links nixpkgs' OpenBLAS (Haswell
# kernels); the run loads the blas-bench shell's AVX-512 OpenBLAS in its place, the stronger
# LAPACK on AVX-512 CPUs. Extra arguments go to the driver, e.g.
# `just bench-linalg --threads 1 --sizes 256,1024`.
bench-linalg *ARGS:
    {{ NIX_DEVELOP }} .#blas-bench --command bash -c "cmake -S benchmarks -B build/benchmarks-lapack -DCMAKE_BUILD_TYPE=Release -DDATAMUNGE_USE_LAPACK=ON -DCMAKE_MAKE_PROGRAM=$(command -v make)"
    {{ NIX_DEVELOP }} .#blas-bench --command bash -c "cmake --build build/benchmarks-lapack --config Release --target linalg_compare -j{{ JOBS }}"
    mkdir -p benchmarks/results/blas
    {{ NIX_DEVELOP }} .#blas-bench --command bash -c 'lib=$(printf "%s" "$DATAMUNGE_BLAS_LIBS" | tr ";" "\n" | sed -n "s|^openblas-avx512=\(.*\)/libopenblas.so.*|\1|p"); LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ./build/benchmarks-lapack/linalg_compare --json benchmarks/results/blas/linalg.json "$@"' linalg_compare {{ ARGS }}
    python3 benchmarks/blas/summarize.py benchmarks/results/blas/linalg.json --out benchmarks/results/blas/linalg.md

# Build and run the datamunge::nn speed benchmarks (GEMM, conv, pooling, batch norm, a cnn_mnist epoch).
bench-nn *ARGS:
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$(command -v make)"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/benchmarks --config Release --target nn_bench -j{{ JOBS }}"
    mkdir -p benchmarks/results
    python3 scripts/benchmark_backend.py run --no-build --suite nn_bench -- {{ ARGS }}

# Persistent observations and raw-run import; history emits TSV for plotting.
benchmark-history *ARGS:
    python3 scripts/benchmark_backend.py history {{ ARGS }}

benchmark-record FILE *ARGS:
    python3 scripts/benchmark_backend.py record '{{ FILE }}' {{ ARGS }}

benchmark-compare BASE NEW:
    python3 scripts/benchmark_backend.py compare '{{ BASE }}' '{{ NEW }}'

# Focused backend suites; forward Google Benchmark options after --.
bench-backend *ARGS:
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/benchmark_backend.py run --jobs {{ JOBS }} {{ ARGS }}

bench-backend-smoke:
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/benchmark_backend.py run --jobs {{ JOBS }} --smoke

# Check the agreed operation matrix and all current tensor optimizer registrations.
bench-backend-audit:
    python3 scripts/benchmark_backend.py audit

# Optional isolated TensorFlow CPU comparison (see benchmarks/TENSORFLOW.md).
bench-tensorflow *ARGS:
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/benchmarks --config Release --target backend_bench -j {{ JOBS }}
    build/tensorflow-venv/bin/python scripts/benchmark_tensorflow.py {{ ARGS }}

# Existing correctness tests linked to the same core as the benchmarks.
bench-backend-check:
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release -DDATAMUNGE_BENCH_CHECKS=ON
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/benchmarks --config Release --target backend_checks -j {{ JOBS }}
    ./build/benchmarks/backend_checks
    python3 -m unittest discover -s benchmarks -p 'test*runner.py'



memcheck:
  valgrind --leak-check=full --track-origins=yes ./build/debug/examples/{{TARGET}}


coverage:
    rm -rf build/coverage
    @bash -c 'set -euo pipefail; \
      gtest_prefix=""; \
      for p in /nix/store/*-gtest-*-dev; do \
        if [ -f "$p/lib/cmake/GTest/GTestConfig.cmake" ]; then gtest_prefix="$p"; break; fi; \
      done; \
      cmake -S tests -B build/coverage/tests -DCMAKE_BUILD_TYPE=Debug -DENABLE_TEST_COVERAGE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_MAKE_PROGRAM="$(command -v make)" ${gtest_prefix:+-DCMAKE_PREFIX_PATH="$gtest_prefix"}; \
      cmake --build build/coverage/tests -j{{ JOBS }} --verbose; \
      ctest --test-dir build/coverage/tests --output-on-failure'
    mkdir -p build/coverage
    lcov -c --rc lcov_function_coverage=0 --ignore-errors mismatch,inconsistent --directory build/coverage/tests --output-file build/coverage/coverage.info
    lcov --ignore-errors inconsistent --ignore-errors unused -r build/coverage/coverage.info '/nix/store/*' --output-file build/coverage/coverage.info
    lcov --ignore-errors inconsistent --ignore-errors unused -r build/coverage/coverage.info '*/_deps/*' --output-file build/coverage/coverage.info
    genhtml --rc genhtml_function_coverage=0 --ignore-errors inconsistent --ignore-errors corrupt build/coverage/coverage.info --output-directory build/coverage/html

test-coverage: coverage
    @bash -c 'set -euo pipefail; \
      info="build/coverage/coverage.info"; \
      test -f "$info"; \
      awk -v want1="$(pwd)/src/datamunge/datamunge.cpp" -v want2="$(pwd)/src/datamunge/datamunge_c.cpp" '\'' \
        function finish() { \
          if (!in_wanted) return; \
          if (total == 0) { \
            printf("coverage: %s: no line data found\\n", file) > "/dev/stderr"; \
            exit 2; \
          } \
          if (hit != total) { \
            printf("coverage: %s: %d/%d lines (%.2f%%)\\n", file, hit, total, (100.0*hit/total)) > "/dev/stderr"; \
            exit 1; \
          } \
        } \
        /^SF:/ { \
          finish(); \
          file = substr($0, 4); \
          in_wanted = (file == want1 || file == want2); \
          total = 0; hit = 0; \
          next; \
        } \
        /^DA:/ { \
          if (!in_wanted) next; \
          split(substr($0, 4), a, ","); \
          total++; \
          if (a[2] + 0 > 0) hit++; \
          next; \
        } \
        /^end_of_record/ { finish(); in_wanted=0; next } \
        END { finish(); print "coverage: OK (datamunge.cpp and datamunge_c.cpp are 100%)" } \
      '\'' "$info"'



debuggable:
  {{ NIX_DEVELOP }} .#cpp --command bash -c 'clang++ -g -O0 debug.cpp -o debug $(pkg-config --cflags datamunge) $(pkg-config --libs-only-L datamunge) $(pkg-config --cflags libxml-2.0) $(pkg-config --libs-only-L libxml-2.0) -std=c++20 -ldatamunge -lxml2'


clean:
    rm -rf build/
    rm -rf {{ BINDINGS_DIR }}/datamungecsharp/bin
    rm -rf {{ BINDINGS_DIR }}/datamungecsharp/obj


# }}} utilities

# {{{ docs commands

org-export INPUT OUTPUT:
    @echo "Exporting {{ INPUT }} -> {{ OUTPUT }}"
    @bash -c 'set -euo pipefail; \
      export_cmd='\''DATAMUNGE_SKIP_DIRENV=1 emacs --batch -Q -l init.el -- "{{ INPUT }}" "{{ OUTPUT }}"'\''; \
      if command -v nix >/dev/null 2>&1; then \
        nix develop --accept-flake-config .#org-examples --command bash -c "$export_cmd"; \
      else \
        bash -c "$export_cmd"; \
      fi'

org-example NAME="dense_linear_algebra" FORMAT="html":
    just org-export examples/org/{{ NAME }}.org build/org/{{ NAME }}.{{ FORMAT }}

# Build every examples/org/*.org file to HTML and PDF under build/org/.
org-examples:
    @echo "Exporting Org examples -> HTML and PDF"
    @bash -c 'set -euo pipefail; \
      export_cmd='\''set -euo pipefail; shopt -s nullglob; files=(examples/org/*.org); if ((${#files[@]} == 0)); then echo "No Org examples found under examples/org/" >&2; exit 1; fi; mkdir -p build/org; for f in "${files[@]}"; do base="$(basename "$f" .org)"; for format in html pdf; do out="build/org/${base}.${format}"; echo "Exporting $f -> $out"; DATAMUNGE_SKIP_DIRENV=1 emacs --batch -Q -l init.el -- "$f" "$out"; done; done'\''; \
      if command -v nix >/dev/null 2>&1; then \
        nix develop --accept-flake-config .#org-examples --command bash -c "$export_cmd"; \
      else \
        bash -c "$export_cmd"; \
      fi'

# Regenerate examples/mepml/*.mepml from examples/org/*.org (needs mep-mepml on PATH),
# compiling and running each report's C++ block to fill in its results.
mepml-from-org:
    python3 scripts/org_to_mepml.py --run examples/org examples/mepml

# Refresh the source catalogue without executing reports.
docs-report-catalogue:
    python3 scripts/website/catalogue.py

# Run every code block of the docs' mepml reports (the documentation collection, or
# FILES) with mep (the flake's, from the cpp shell), as its C-c C-c does, writing
# each block's results and figures back into its report. The C++ blocks build against
# datamunge found by pkg-config. Prints each block's outcome and fails if one failed
# (a failed block keeps its old results). `just docs` runs only the changed reports.
docs-mepml-run *FILES:
    #!/usr/bin/env bash
    set -euo pipefail
    files=({{ if FILES == "" { "$(find docs/mepml -name '*.mepml' | sort)" } else { FILES } }})
    pc="{{ MEPML_PKG_CONFIG_PATH }}"
    # mep finds its own libraries; an outside LD_LIBRARY_PATH can stop it starting.
    {{ NIX_DEVELOP }} .#cpp --command env -u LD_LIBRARY_PATH ${pc:+PKG_CONFIG_PATH="$pc"} \
      bash -c 'python3 scripts/website/reports.py --check-toolchain "$@" && exec mep-mepml run --timeout 1800 "$@"' _ "${files[@]}"

# The website, as `just docs` builds it (kept for its older name).
docs-mepml *ARGS: (docs ARGS)

# Bump the flake's mep (the editor and `mep-mepml`, which the docs build uses) to the
# latest commit of its main branch.
update-mep:
    nix flake update mep

prebuild-docs-pages:
    @echo "Exporting Org pages -> Markdown"
    @bash -c 'set -euo pipefail; \
      export_cmd='\''set -euo pipefail; shopt -s nullglob; for f in docs/org/pages/*.org; do base="$(basename "$f" .org)"; out="docs/pages/${base}.md"; emacs --batch -Q -l docs/org-to-md.el -- "$f" "$out"; done'\''; \
      if command -v nix >/dev/null 2>&1; then \
        if nix develop --accept-flake-config .#docs-pages --command bash -c "$export_cmd" >/dev/null 2>&1; then \
          nix develop --accept-flake-config .#docs-pages --command bash -c "$export_cmd"; \
        else \
          bash -c "$export_cmd"; \
        fi; \
      else \
        bash -c "$export_cmd"; \
      fi'

# Convenience alias: build every docs/org/pages/*.org page to Markdown (docs/pages/*.md).
docs-md: prebuild-docs-pages
    @echo "Org pages exported to Markdown under docs/pages/"

# Export a single arbitrary Org file to Markdown, executing Babel blocks best-effort
# (failures are captured into an "Export notes" section instead of aborting the export).
docs-md-file INPUT OUTPUT:
    @echo "Exporting {{ INPUT }} -> {{ OUTPUT }} (Markdown, best-effort Babel)"
    @bash -c 'set -euo pipefail; \
      export_cmd='\''emacs --batch -Q -l docs/org-to-md.el -- "{{ INPUT }}" "{{ OUTPUT }}"'\''; \
      if command -v nix >/dev/null 2>&1; then \
        nix develop --accept-flake-config .#docs-pages --command bash -c "$export_cmd"; \
      else \
        bash -c "$export_cmd"; \
      fi'

# Build the documentation website in build/debug/docs/site: Doxygen reads the headers
# into XML, mep runs the mepml reports that changed since their last run (and every one
# that failed) and exports them, and scripts/website/build.py writes the site, with a
# "Report build status" page. Fails if a report's code block failed. ARGS go to CMake:
# -DDOCS_MEPML_FORCE=ON reruns every report, -DDOCS_MEPML_JOBS=N runs N reports at once
# (default: one per core, as memory allows), -DDOCS_MEPML_PCH=OFF compiles the blocks
# with plain g++ (not clang++ with a precompiled dm.hpp), -DDOCS_MEPML_RUN=OFF uses the
# stored results, -DDOCS_MEPML_ALLOW_FAILURES=ON builds the site anyway.
docs *ARGS:
    @echo "Building docs"
    {{ NIX_DEVELOP }} .#cpp --command cmake -U MEP_MEPML_EXECUTABLE -S docs -B build/debug/docs -DDOCS_MEPML_PKG_CONFIG_PATH="{{ MEPML_PKG_CONFIG_PATH }}" -DDOCS_MEPML_RUN=ON -DDOCS_MEPML_ALLOW_FAILURES=OFF -DDOCS_MEPML_FORCE=OFF {{ ARGS }}
    {{ NIX_DEVELOP }} .#cpp --command cmake --build build/debug/docs --target Website

# Rerun every report and refresh its cached code chunk results.
docs-force *ARGS: (docs ARGS + " -DDOCS_MEPML_FORCE=ON")

# Export stored report results and the API reference without running experiments.
docs-preview:
    @bash -c 'set -euo pipefail; \
      tools=(); \
      if ! command -v doxygen >/dev/null || ! command -v mep-mepml >/dev/null || \
         ! python3 -c "import jinja2, markdown_it, pygments" 2>/dev/null; then \
        tools=({{ NIX_DEVELOP }} .#cpp-tools --command); \
      fi; \
      "${tools[@]}" cmake -U MEP_MEPML_EXECUTABLE -S docs -B build/debug/docs -DDOCS_MEPML_RUN=OFF -DDOCS_MEPML_ALLOW_FAILURES=ON; \
      "${tools[@]}" cmake --build build/debug/docs --target Website'

# Publish a local export of the saved report results and figures.
docs-publish: docs-preview
    bash scripts/publish_docs_branch.sh build/debug/docs/site

# Serve the built website at http://localhost:PORT (it also works opened from disk).
docs-serve PORT="8000":
    @test -f build/debug/docs/site/index.html || { echo "No website yet: run 'just docs' first" >&2; exit 1; }
    @echo "Serving the website at http://localhost:{{ PORT }}/"
    python3 -m http.server {{ PORT }} --bind 127.0.0.1 -d build/debug/docs/site

# Embed data-raw/datasets/*.csv into src/datamunge/datasets/embedded/ (the built-in datasets).
generate-datasets:
    python3 -I scripts/generate_datasets.py

# Inventory every maintained core definition, including private/static internals.
# Add --check and repeatable --path arguments to enforce a completed source batch.
# Regenerate include/datamunge/dir_registry.hpp (the member tables behind dm::dir()) from fresh Doxygen XML.
generate-dir:
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S docs -B build/debug/docs
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/debug/docs --target GenerateDocsXml
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/generate_dir_registry.py

docs-audit *ARGS:
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S docs -B build/debug/docs
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/debug/docs --target GenerateDocsXml
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/check_documentation.py --json build/debug/docs/coverage.json {{ ARGS }}

# Check descriptions returned by clangd at representative source definitions.
docs-hover *ARGS:
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/check_documentation_hover.py {{ ARGS }}

# Native library simulation tests; optional GPU tests report unavailable backends explicitly.
# Opt-in SDK build; the host NVIDIA driver is discovered at runtime, including NixOS.
build-cuda *CMAKE_ARGS:
    {{ NIX_DEVELOP }} .#cuda --command cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release -DDATAMUNGE_ENABLE_CUDA=ON {{ CMAKE_ARGS }}
    {{ NIX_DEVELOP }} .#cuda --command cmake --build build/cuda --target datamunge -j{{ JOBS }}

test-cuda *CMAKE_ARGS:
    {{ NIX_DEVELOP }} .#cuda --command cmake -S tests -B build/cuda-tests -DCMAKE_BUILD_TYPE=Release -DDATAMUNGE_ENABLE_CUDA=ON {{ CMAKE_ARGS }}
    {{ NIX_DEVELOP }} .#cuda --command cmake --build build/cuda-tests --target run_simulation_tests run_nn_device_tests check_simulation_allocations -j{{ JOBS }}
    {{ NIX_DEVELOP }} .#cuda --command env DATAMUNGE_REQUIRE_CUDA=1 ctest --test-dir build/cuda-tests -R '^(simulation|nn-device)\.' --output-on-failure

test-cad *CMAKE_ARGS:
  {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S tests/cad -B build/cad-tests -DCMAKE_BUILD_TYPE=Release {{ CMAKE_ARGS }}
  {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/cad-tests -j{{ JOBS }}
  {{ NIX_DEVELOP }} .#cpp-tools --command ctest --test-dir build/cad-tests --output-on-failure

cad-assets:
  python3 scripts/fetch_cad_example.py

cad-rebot: cad-assets
  mkdir -p build/cad-demo
  {{ NIX_DEVELOP }} .#cpp --command bash -c 'c++ -std=c++23 -O2 examples/cpp/source/cad/rebot_arm.cpp -o build/cad-demo/rebot_arm $(pkg-config --cflags --libs datamunge) -Wl,-rpath,$(pkg-config --variable=libdir datamunge) && build/cad-demo/rebot_arm'

test-simulation *CMAKE_ARGS:
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S tests -B build/simulation -DCMAKE_BUILD_TYPE=Release {{ CMAKE_ARGS }}
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/simulation --target run_simulation_tests check_simulation_allocations -j{{ JOBS }}
    {{ NIX_DEVELOP }} .#cpp-tools --command ctest --test-dir build/simulation -R '^simulation\.' --output-on-failure

# Regenerate the native Vulkan shader test asset with a standard SDK shader compiler.
prebuild-compute-fixtures *ARGS:
    python3 scripts/compile_compute_fixture.py {{ ARGS }}

prebuild-simulation-shaders *ARGS:
    python3 scripts/prebuild_simulation_shaders.py {{ ARGS }}

# Native tensor storage shaders; generated SPIR-V is consumed without an SDK.
prebuild-nn-storage-shaders *ARGS:
    python3 scripts/prebuild_nn_storage_shaders.py {{ ARGS }}

# Native CUDA nn inference kernels (shaders/nn/inference.cu); generated PTX is consumed without nvcc.
# In-house CUDA GEMM (Context::matmul) against cuBLAS (no TF32) at square and SqueezeNet
# shapes, float and double. Writes benchmarks/results/blas/cuda.json and cuda.md.
bench-blas-cuda *ARGS:
    {{ NIX_DEVELOP }} .#cuda --command cmake -S benchmarks -B build/benchmarks-cuda -DCMAKE_BUILD_TYPE=Release -DDATAMUNGE_ENABLE_CUDA=ON -DDATAMUNGE_ENABLE_CUBLAS=ON
    {{ NIX_DEVELOP }} .#cuda --command cmake --build build/benchmarks-cuda --target cuda_gemm_compare -j{{ JOBS }}
    mkdir -p benchmarks/results/blas
    {{ NIX_DEVELOP }} .#cuda --command ./build/benchmarks-cuda/cuda_gemm_compare --json benchmarks/results/blas/cuda.json {{ ARGS }}
    python3 benchmarks/blas/summarize.py benchmarks/results/blas/cuda.json --out benchmarks/results/blas/cuda.md

# CAD ray tracer only; ordinary library builds need no CUDA SDK.
prebuild-cad-cuda-kernels:
    {{ NIX_DEVELOP }} .#cuda --command python3 scripts/prebuild_nn_cuda_kernels.py --only render

prebuild-nn-cuda-kernels *ARGS:
    {{ NIX_DEVELOP }} .#cuda --command python3 scripts/prebuild_nn_cuda_kernels.py {{ ARGS }}

test-nn-device *CMAKE_ARGS:
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S tests -B build/simulation -DCMAKE_BUILD_TYPE=Release {{ CMAKE_ARGS }}
    {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/simulation --target run_nn_device_tests -j{{ JOBS }}
    {{ NIX_DEVELOP }} .#cpp-tools --command ctest --test-dir build/simulation -R '^nn-device\.' --output-on-failure

test-simulation-examples: test-simulation
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 scripts/check_simulation_examples.py --library-dir build/simulation/_deps/datamunge-build

# Real Doxygen fixtures verify that incomplete or stale inventories cannot pass.
test-docs:
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 -m unittest discover -s tests/python -p test_documentation_audit.py -v
    {{ NIX_DEVELOP }} .#cpp-tools --command python3 -m pytest -q tests/python/test_website_toolchain.py tests/python/test_website_model_diagrams.py

docs-bindings: prebuild-swig
    @echo "Binding documentation is emitted as SWIG-generated docstrings/comments (per language)."

docs-all: docs docs-bindings
    @echo "Core + binding docs are up to date."

# Regenerate README.md from its editable MepML source.
readme MEP_MEPML="mep-mepml":
    {{ MEP_MEPML }} build README.mepml md

# }}} docs commands

# {{{ pdf commands
#
# PDF export reuses org-export's generic init.el pipeline (backend inferred from the
# .pdf extension). SVG figures can't be rasterized by pdflatex directly, so init.el
# converts every \includegraphics{...svg} to a sibling PDF via rsvg-convert on export;
# both pdflatex and rsvg-convert live in the .#docs-pages devShell.

# Build one examples/org/*.org file to PDF, e.g. `just org-pdf lda_iris_analysis`.
org-pdf NAME="dense_linear_algebra":
    just org-export examples/org/{{ NAME }}.org build/org/{{ NAME }}.pdf

# Build every docs/org/pages/*.org page to PDF under build/org/pages-pdf/.
docs-pdf:
    @echo "Exporting Org pages -> PDF"
    @bash -c 'set -euo pipefail; \
      export_cmd='\''set -euo pipefail; shopt -s nullglob; mkdir -p build/org/pages-pdf; for f in docs/org/pages/*.org; do base="$(basename "$f" .org)"; out="build/org/pages-pdf/${base}.pdf"; emacs --batch -Q -l init.el -- "$f" "$out"; done'\''; \
      if command -v nix >/dev/null 2>&1; then \
        nix develop --accept-flake-config .#docs-pages --command bash -c "$export_cmd"; \
      else \
        bash -c "$export_cmd"; \
      fi'

# Open a PDF with the user's preferred viewer: $BROWSER, then the first common
# viewer found on PATH.
view-pdf PATH:
    @bash -c 'set -euo pipefail; \
      f="{{ PATH }}"; \
      if [ ! -f "$f" ]; then echo "No such file: $f" >&2; exit 1; fi; \
      if [ -n "${BROWSER:-}" ]; then exec "$BROWSER" "$f"; fi; \
      for viewer in xdg-open zathura evince okular mupdf open; do \
        if command -v "$viewer" >/dev/null 2>&1; then exec "$viewer" "$f"; fi; \
      done; \
      echo "No PDF viewer found on PATH; open $f manually." >&2; exit 1'

# Build then open a single example's PDF, e.g. `just view-org-pdf lda_iris_analysis`.
view-org-pdf NAME="dense_linear_algebra": (org-pdf NAME)
    just view-pdf build/org/{{ NAME }}.pdf

# }}} pdf commands

# {{{ example commands

examples:
    @echo "Building Examples"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake -S examples/cpp -B build/debug/examples --preset=debug -DCMAKE_MAKE_PROGRAM=$(command -v make) -DBUILD_W_INSTALLED=OFF"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "cmake --build build/debug/examples -j{{ JOBS }} --verbose"
    {{ NIX_DEVELOP }} .#cpp --command bash -c "find ./build -name 'compile_commands.json' -exec cat {} + | jq -s add > compile_commands.json"
    # nix develop . --command bash -c "make -C ./build/debug/examples -j10 --verbose"

example EXAMPLE:
    @echo "Running Example {{ EXAMPLE }}:"
    ./build/debug/examples/{{ EXAMPLE }}

example-python:
    {{ NIX_DEVELOP }} .#python --command bash -c "python examples/python/datamunge_ex.py"

# }}} example commands

# {{{ view commands

view-flamechart:
    $(BROWSER) ./flamegraph.svg

view-docs:
    @bash -c 'set -euo pipefail; \
      page="$PWD/build/debug/docs/site/index.html"; \
      if [[ ! -f "$page" ]]; then just docs-preview; fi; \
      if [[ -n "${BROWSER:-}" ]]; then exec "$BROWSER" "$page"; fi; \
      for opener in xdg-open open; do \
        if command -v "$opener" >/dev/null 2>&1; then exec "$opener" "$page"; fi; \
      done; \
      echo "Set BROWSER or open $page manually." >&2; exit 1'

# }}} view commands

# {{{ windows specifics

windows-run: build
    @echo "Running target ${TARGET}"
    build/debug/examples/${TARGET}

# }}} windows specifics

# {{{ all commands

test-nn-model-plot:
  {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S tests -B build/debug/tests -DCMAKE_BUILD_TYPE=Debug
  {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/debug/tests --target run_nn_model_plot_tests -j{{ NN_JOBS }}
  {{ NIX_DEVELOP }} .#cpp-tools --command ctest --test-dir build/debug/tests -R '^nn-model-plot\.' --output-on-failure --no-tests=error

test-nn-metrics-plot:
  {{ NIX_DEVELOP }} .#cpp-tools --command cmake -S tests -B build/debug/tests -DCMAKE_BUILD_TYPE=Debug
  {{ NIX_DEVELOP }} .#cpp-tools --command cmake --build build/debug/tests --target run_nn_metrics_plot_tests -j{{ NN_JOBS }}
  {{ NIX_DEVELOP }} .#cpp-tools --command ctest --test-dir build/debug/tests -R '^nn-metrics-plot\.' --output-on-failure --no-tests=error

test-nix:
  nix flake check --accept-flake-config -L{{ SWIG_OVERRIDE }}

build-nix PACKAGE="datamunge":
  nix build --accept-flake-config{{ SWIG_OVERRIDE }} ".#{{ PACKAGE }}"

all-test:
  bash ./scripts/test_all.sh

test-all: all-test

# }}} all commands
